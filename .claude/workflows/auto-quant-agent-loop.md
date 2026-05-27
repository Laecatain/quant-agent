# Auto Quant Agent Multi-Agent 工程工作流

**Last Updated:** 2026-05-27  
**适用项目:** Python 量化因子 agent（`auto_quant_project/`）

## 触发场景

当任务涉及 FactorMiner、sandbox/evaluator/scoring/splitter、实验存储、数据下载或主循环集成等实质性工程变更时，使用本工作流。目标是把“规划、红灯测试、最小实现、验证、审查、提交”做成可重复闭环，并避免在权限受限或审查未通过时盲目继续写代码。

不适用场景：纯文档补充、只读代码调研、单行配置调整。此类任务可直接完成，但仍需按需运行相关验证。

## Agent 编排

1. **并行规划**：同时启动 `architect`、`planner`、`tdd-guide`。
   - `architect` 输出模块边界、数据流和集成风险。
   - `planner` 拆解实施步骤、依赖和回滚点。
   - `tdd-guide` 定义红灯测试：输入、预期失败、边界条件。
2. **红灯测试优先**：TDD agent 先新增或修改 pytest，主 agent 只运行目标测试并确认失败原因与需求匹配。
3. **最小生产实现**：代码生成 agent 只实现让红灯测试转绿所需的最小代码，优先保持现有索引对齐、无未来函数、显式错误处理。
4. **主 agent 验证**：主 agent 运行 pytest；若失败，调用修复 agent 根据错误增量修复，不扩大范围。
5. **并行审查**：测试通过后并行运行 python/code/security reviewers。出现 CRITICAL 或 HIGH 阻塞问题时，回到修复与测试阶段。
6. **自动提交**：审查通过后自动进入 git 提交流程。只暂存本轮自洽文件集合，提交信息写清楚本轮能力增量、验证结果和未纳入范围；提交若被权限分类器挂起，停止 git 操作，保留当前状态，转入下一轮只读规划并在权限恢复后重试。

## 并行与串行边界

可并行：只读架构分析、实现方案比较、测试用例设计、代码审查、安全审查、性能审查、测试分片、依赖影响分析。

必须串行：修改同一文件、失败修复、格式化后验证、sandbox/evaluator 对齐逻辑调整、数据库或文件落盘语义变更、git add/commit/push。串行阶段以“一个明确失败原因、一个最小修复、一次验证”为单位推进。

## 验收命令

优先运行与变更最接近的测试：

```bash
python -m pytest path/to/test_file.py -q
python -m pytest path/to/test_file.py::test_name -q
```

涉及 sandbox/evaluator/factor 对齐时，增加基准验证：

```bash
python auto_quant_project/scripts/run_full_sandbox_benchmark.py --limit-codes 10
python auto_quant_project/scripts/run_full_sandbox_benchmark.py
```

提交前至少查看：

```bash
git status
git diff
```

若项目使用本地 venv，使用 `auto_quant_project/.venv/Scripts/python.exe -m pytest ...`；在 macOS/Linux 使用 `auto_quant_project/.venv/bin/python -m pytest ...`。

## 失败处理

- 红灯未失败：测试无效，先修测试，不写生产代码。
- 测试失败但原因不匹配：回到 TDD agent 澄清断言或 fixture。
- 实现后仍失败：记录最小错误输出，调用修复 agent；禁止顺手重构无关模块。
- 审查阻塞：先修 CRITICAL/HIGH，再重跑目标测试和对应 reviewer。
- 安全问题：停止提交，检查同类问题；涉及密钥时要求轮换。
- 权限分类器阻塞提交：不绕过权限，不使用 destructive git；转为只读复盘和下一轮计划。

## 自动提交规则

本工作流默认在测试和 CRITICAL/HIGH 审查通过后自动提交，不再等待额外确认。提交仍必须遵守安全边界：不自动 push，不 force push，不 amend，不跳过 hooks，不提交密钥、大型数据或生成结果。

提交前精确暂存本轮自洽文件集合，避免把用户已有无关改动混入。提交信息使用 conventional commits，并在正文描述：

1. 本轮新增能力。
2. 已运行的验证命令和结果。
3. 明确未纳入的无关文件或生成物。

示例：

```text
feat: add factor backtesting loop

Add StrategySpec/backtest_factor_strategy and record backtest summaries in successful FactorMiner trials. Verified with focused and full pytest runs; unrelated generated artifacts were left unstaged.
```

## 下一轮循环模板

下一轮默认从“FactorMiner 集成”开始：

1. 并行规划 FactorMiner 与 `storage.ExperimentStore`、scoring、splitter 的边界。
2. 写红灯 pytest 覆盖试验落盘、best factors 更新、失败 trial 记录。
3. 最小实现集成点。
4. 跑目标 pytest，再按需跑 sandbox benchmark。
5. 并行 python/code/security review。
6. 通过后自动提交；若提交受阻，记录待提交文件集合并进入只读规划下一轮。
