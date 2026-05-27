"""
Static safety checker for LLM-generated alpha factor code.

The checker is intentionally conservative about Python runtime escape hatches while
allowing common vectorized pandas/numpy factor expressions such as groupby,
rolling, rank, pct_change and transform. It is a pre-execution guard, not a full
security sandbox; runtime isolation is still required for defense in depth.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CheckResult:
    """Result returned by :func:`check_factor_code`.

    Attributes:
        passed: True when no blocking safety errors were found.
        errors: Blocking issues. Code with errors must not be executed.
        warnings: Non-blocking suspicious patterns or quality hints.
    """

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Modules/packages that should never be reachable from factor code. The sandbox
# only intends to expose data, pd and np; these names are blocked both as import
# targets and as direct global/object attribute access roots.
_FORBIDDEN_MODULE_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "pathlib",
    "shutil",
    "glob",
    "pickle",
    "marshal",
    "shelve",
    "sqlite3",
    "importlib",
    "builtins",
    "inspect",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
    "tempfile",
    "webbrowser",
}

_FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "memoryview",
}

_FORBIDDEN_METHODS = {
    # File IO / path traversal.
    "read",
    "read_text",
    "read_bytes",
    "readlines",
    "write",
    "write_text",
    "write_bytes",
    "writelines",
    "rename",
    "replace",
    "unlink",
    "remove",
    "rmdir",
    "mkdir",
    "makedirs",
    "touch",
    "chmod",
    "chown",
    "stat",
    "lstat",
    "iterdir",
    "glob",
    "rglob",
    "resolve",
    "absolute",
    "expanduser",
    # Process / shell execution.
    "system",
    "popen",
    "run",
    "call",
    "check_call",
    "check_output",
    "spawn",
    "fork",
    "kill",
    "execv",
    "execve",
    # Network requests.
    "request",
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "head",
    "options",
    "urlopen",
    "urlretrieve",
    "connect",
    "bind",
    "listen",
    "accept",
    "send",
    "sendall",
    "recv",
    "recvfrom",
}

_FORBIDDEN_ATTRIBUTES = {
    "__class__",
    "__bases__",
    "__base__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
    "__dict__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
}

# Regex fallback catches syntax-obfuscated or otherwise suspicious strings even
# when AST parsing fails. Keep pandas-safe methods out of this list.
_STRING_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(^|\n)\s*(import|from)\s+", re.IGNORECASE), "禁止使用 import/from import。"),
    (re.compile(r"\b(open|eval|exec|compile|__import__)\s*\(", re.IGNORECASE), "禁止调用 open/eval/exec/compile/__import__。"),
    (re.compile(r"\b(os|sys|subprocess|socket|requests|urllib|http|pathlib|shutil|importlib|builtins)\b", re.IGNORECASE), "禁止访问系统、文件或网络相关模块。"),
    (re.compile(r"\.\s*(read|read_text|read_bytes|write|write_text|write_bytes|unlink|remove|rename)\s*\(", re.IGNORECASE), "禁止文件读写或文件系统修改。"),
    (re.compile(r"\.\s*(system|popen|run|check_call|check_output)\s*\(", re.IGNORECASE), "禁止进程或 shell 执行。"),
    (re.compile(r"\.\s*(request|get|post|put|patch|delete|urlopen|connect|send|recv)\s*\(", re.IGNORECASE), "禁止网络请求或 socket 通信。"),
    (re.compile(r"\bshift\s*\(\s*-\s*(?:[1-9]\d*|[a-zA-Z_])", re.IGNORECASE), "禁止使用 shift 负值，疑似未来函数。"),
    (re.compile(r"__\s*(?:class|bases|base|subclasses|mro|globals|code|closure|dict|getattribute)\s*__", re.IGNORECASE), "禁止反射/逃逸相关双下划线属性。"),
    (re.compile(r"\bto_(?:csv|excel|parquet|pickle|feather|hdf|json|sql)\s*\(", re.IGNORECASE), "禁止写出文件或外部存储。"),
    (re.compile(r"\bread_(?:csv|excel|parquet|pickle|feather|hdf|json|sql|table)\s*\(", re.IGNORECASE), "禁止从文件或外部存储读取数据。"),
)

_PANDAS_SAFE_METHODS = {
    "groupby",
    "rolling",
    "rank",
    "pct_change",
    "transform",
    "mean",
    "std",
    "sum",
    "min",
    "max",
    "median",
    "quantile",
    "diff",
    "shift",
    "fillna",
    "replace",
    "clip",
    "where",
    "mask",
    "reindex",
    "reset_index",
    "set_index",
    "sort_values",
    "copy",
    "astype",
    "apply",
    "map",
    "abs",
    "pow",
    "stack",
    "unstack",
    "pivot",
    "pivot_table",
    "merge",
    "join",
    "dropna",
    "isna",
    "notna",
}


def check_factor_code(code: str) -> CheckResult:
    """Check generated factor code before sandbox execution.

    Args:
        code: Python source code generated by an LLM. The intended output is a
            pandas.Series variable named ``factor``.

    Returns:
        CheckResult: ``passed`` is False when blocking errors are detected.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(code, str):
        return CheckResult(passed=False, errors=["code 必须是字符串。"], warnings=[])

    stripped = code.strip()
    if not stripped:
        return CheckResult(passed=False, errors=["代码为空。"], warnings=[])

    _run_string_fallback(stripped, errors)

    try:
        tree = ast.parse(stripped)
    except SyntaxError as exc:
        errors.append(f"代码存在语法错误：{exc.msg} (line {exc.lineno})。")
        return _dedupe_result(errors, warnings)

    visitor = _SafetyVisitor(errors=errors, warnings=warnings)
    visitor.visit(tree)

    if not _assigns_factor(tree):
        warnings.append("代码未显式赋值变量 `factor`，沙盒执行时会失败。")

    return _dedupe_result(errors, warnings)


class _SafetyVisitor(ast.NodeVisitor):
    """AST visitor that records blocking safety errors and suspicious warnings."""

    def __init__(self, errors: list[str], warnings: list[str]) -> None:
        self.errors = errors
        self.warnings = warnings

    def visit_While(self, node: ast.While) -> None:  # noqa: N802 - ast API
        self.warnings.append("检测到显式 while 循环；因子代码建议使用 pandas/numpy 向量化实现。")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802 - ast API
        if _is_row_iteration(node.iter):
            self.errors.append("禁止逐行遍历 data.iterrows()/data.itertuples()，请改用向量化 pandas 操作。")
        else:
            self.warnings.append("检测到显式 for 循环；因子代码建议使用 pandas/numpy 向量化实现。")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast API
        names = ", ".join(alias.name for alias in node.names)
        self.errors.append(f"禁止使用 import 语句：{names}。")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast API
        module = node.module or ""
        self.errors.append(f"禁止使用 from import 语句：{module}。")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 - ast API
        if node.id in _FORBIDDEN_MODULE_ROOTS:
            self.errors.append(f"禁止访问高风险模块或对象：{node.id}。")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - ast API
        attr = node.attr
        root = _root_name(node)
        dotted = _dotted_name(node)

        if attr in _FORBIDDEN_ATTRIBUTES or (attr.startswith("__") and attr.endswith("__")):
            self.errors.append(f"禁止反射/逃逸属性访问：{dotted or attr}。")
        elif root in _FORBIDDEN_MODULE_ROOTS:
            self.errors.append(f"禁止访问高风险模块属性：{dotted or root}。")
        elif attr in _FORBIDDEN_METHODS and attr not in _PANDAS_SAFE_METHODS:
            self.errors.append(f"禁止访问高风险方法或属性：{dotted or attr}。")

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        func_name = _call_name(node.func)
        root = _root_name(node.func)

        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            self.errors.append(f"禁止调用高风险内置函数：{node.func.id}()。")
        elif root in _FORBIDDEN_MODULE_ROOTS:
            self.errors.append(f"禁止调用高风险模块接口：{func_name or root}()。")
        elif isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            dotted = _dotted_name(node.func)
            if attr in _FORBIDDEN_METHODS and attr not in _PANDAS_SAFE_METHODS:
                self.errors.append(f"禁止调用高风险方法：{dotted or attr}()。")
            if attr == "shift" and _has_negative_shift(node):
                self.errors.append("禁止使用 shift 负值，疑似未来函数。")
            if _is_pandas_io_call(node.func):
                self.errors.append(f"禁止 pandas 文件/外部存储 IO：{dotted or attr}()。")

        # Dynamic reflection via getattr(x, "__subclasses__") or getattr(os, "system").
        if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}:
            reflected = _literal_arg(node, 1)
            if reflected:
                self.errors.append(f"禁止反射访问：{node.func.id}(..., {reflected!r})。")

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802 - ast API
        # globals()['__builtins__'] / obj['__class__'] style escape attempts.
        key = _literal_slice(node.slice)
        if isinstance(key, str) and (
            key in _FORBIDDEN_ATTRIBUTES
            or key in _FORBIDDEN_MODULE_ROOTS
            or (key.startswith("__") and key.endswith("__"))
        ):
            self.errors.append(f"禁止通过下标访问高风险键：{key!r}。")
        self.generic_visit(node)



def _run_string_fallback(code: str, errors: list[str]) -> None:
    for pattern, message in _STRING_RULES:
        if pattern.search(code):
            errors.append(message)



def _is_row_iteration(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in {"iterrows", "itertuples"}:
        return False
    root = _root_name(node.func)
    return root == "data"



def _assigns_factor(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _target_contains_name(target, "factor"):
                    return True
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if _target_contains_name(node.target, "factor"):
                return True
    return False



def _target_contains_name(target: ast.AST, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_contains_name(elt, name) for elt in target.elts)
    return False



def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    return _dotted_name(func)



def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None



def _root_name(node: ast.AST) -> str | None:
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Call):
        return _root_name(current.func)
    if isinstance(current, ast.Subscript):
        return _root_name(current.value)
    if isinstance(current, ast.Name):
        return current.id
    return None



def _has_negative_shift(call: ast.Call) -> bool:
    if call.args and _is_negative_numeric(call.args[0]):
        return True
    for keyword in call.keywords:
        if keyword.arg in {"periods", "period"} and _is_negative_numeric(keyword.value):
            return True
    return False



def _is_negative_numeric(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return node.operand.value > 0
        # shift(-n) is also suspicious even if n is a variable.
        if isinstance(node.operand, ast.Name):
            return True
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value < 0
    return False



def _is_pandas_io_call(func: ast.Attribute) -> bool:
    attr = func.attr
    root = _root_name(func)
    return (root in {"pd", "data"} and attr.startswith("read_")) or attr in {
        "to_csv",
        "to_excel",
        "to_parquet",
        "to_pickle",
        "to_feather",
        "to_hdf",
        "to_json",
        "to_sql",
    }



def _literal_arg(call: ast.Call, index: int) -> str | None:
    if len(call.args) <= index:
        return None
    arg = call.args[index]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None



def _literal_slice(node: ast.AST) -> object | None:
    if isinstance(node, ast.Constant):
        return node.value
    # Python <3.9 compatibility if needed.
    if isinstance(node, ast.Index):  # pragma: no cover[attr-defined]
        return _literal_slice(node.value)  # type: ignore[attr-defined]
    return None



def _dedupe_result(errors: list[str], warnings: list[str]) -> CheckResult:
    unique_errors = _dedupe(errors)
    unique_warnings = _dedupe(warnings)
    return CheckResult(passed=not unique_errors, errors=unique_errors, warnings=unique_warnings)



def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
