from __future__ import annotations

import pytest

from core.static_checker import check_factor_code


@pytest.mark.parametrize(
    "code",
    [
        "factor = np.load('factor.npy')",
        "np.save('factor.npy', data['close'].to_numpy())\nfactor = data['close']",
        "factor = np.fromfile('factor.bin')",
        "factor = np.lib.format.open_memmap('factor.npy')",
        "np.savetxt('factor.txt', data['close'].to_numpy())\nfactor = data['close']",
        "data['close'].to_numpy().tofile('factor.bin')\nfactor = data['close']",
        "data['close'].to_numpy().dump('factor.pkl')\nfactor = data['close']",
        "np.ctypeslib.load_library('native', '.')\nfactor = data['close']",
        "np.lib.npyio.zipfile.ZipFile('factor.zip', 'w').writestr('x', 'y')\nfactor = data['close']",
        "np.lib.npyio.zipfile_factory('factor.zip', 'w').writestr('x', 'y')\nfactor = data['close']",
        "factor = pd.Series(np.lib.npyio.NpzFile('factor.npz')['arr_0'])",
        "factor = pd.Series(np.lib.npyio.DataSource().open('factor.npy', 'rb').read())",
        "factor = pd.Series(np.lib.format.read_array(data['close']))",
        "np.lib.format.write_array(data['close'], data['close'].to_numpy())\nfactor = data['close']",
        "factor = pd.Series(np.recfromcsv('secret.csv'))",
        "factor = pd.Series(np.lib.npyio.recfromtxt('secret.txt'))",
        "s = \"op\" + \"en('pwned.txt','w').wr\" + \"ite('x')\"\nnp.testing.runstring(s, {})\nfactor = data['close']",
        "np.testing.rundocs('factor = data[\\'close\\']')\nfactor = data['close']",
        "np.test(tests=[])\nfactor = data['close']",
        "factor = pd.Series(np._pytesttester.PytestTester('numpy'))",
        "pd.test()\nfactor = data['close']",
        "pd.testing.assert_series_equal(data['close'], data['close'])\nfactor = data['close']",
        "np.lib.format.pickle.loads(b\"cos\\\\nsystem\\\\n(S'echo pwned'\\\\ntR.\")\nfactor = data['close']",
        "handle = pd.io.common.get_handle('factor.txt', 'w').handle\ndata[['close']].to_string(buf=handle)\nfactor = data['close']",
        "data.to_html('pwned.html')\nfactor = data['close']",
        "data.to_xml('pwned.xml')\nfactor = data['close']",
        "data.to_stata('pwned.dta')\nfactor = data['close']",
        "data.to_orc('pwned.orc')\nfactor = data['close']",
        "store = pd.HDFStore('pwned.h5', mode='w')\nfactor = data['close']",
        "data.to_clipboard()\nfactor = data['close']",
        "factor = pd.Series(pd.ExcelFile('secret.xlsx').parse()['close'])",
        "writer = pd.ExcelWriter('pwned.xlsx')\nwriter.close()\nfactor = data['close']",
        "reader = pd.read_csv\nfactor = reader('secret.csv')['close']",
        "factor = pd.read_html('secret.html')[0].iloc[:, 0]",
        "factor = pd.read_xml('secret.xml')['close']",
        "factor = pd.read_clipboard()['close']",
        "p = pd\nfactor = p.read_html('secret.html')[0].iloc[:, 0]",
        "sp = pd.compat._optional.import_optional_dependency('sub' + 'process')\nP = sp.Popen\nP(['noop'])\nfactor = data['close']",
        "getter = pd.core.frame.operator.attrgetter('compat._optional.import_optional_dependency')\nloader = getter(pd)\nsp = loader('sub' + 'process')\nP = pd.core.frame.operator.attrgetter('Po' + 'pen')(sp)\nP(['noop'])\nfactor = data['close']",
        "expr = \"@p\" + \"d.compat._optional.import_optional_dependency('sub'+'pro'+'cess').Po\" + \"pen(['noop'])\"\ndata.query(expr)\nfactor = data['close']",
        "e = data.eval\nexpr = \"@p\" + \"d.compat._optional.import_optional_dependency('sub'+'process').check_output(['noop'])\"\nfactor = pd.Series([e(expr)] * len(data), index=data.index)",
        "ax = data[['close']].plot()\nax.get_figure().savefig('pwned.png')\nfactor = data['close']",
        "pd.show_versions(as_json='pwned.json')\nfactor = data['close']",
        "axes = pd.plotting.scatter_matrix(data[['open', 'close']])\naxes[0, 0].figure.canvas.print_png('pwned.png')\nfactor = data['close']",
        "writer = data.to_csv\nwriter('pwned.csv')\nfactor = data['close']",
        "s = data['close'] + 1\nwriter = s.to_csv\nwriter('pwned.csv')\nfactor = data['close']",
    ],
)
def test_rejects_numpy_file_io(code: str) -> None:
    result = check_factor_code(code)

    assert result.passed is False
    assert any(
        "NumPy" in error
        or "文件" in error
        or "高风险" in error
        or "包测试" in error
        or "进程" in error
        or "eval" in error
        for error in result.errors
    )


def test_allows_vectorized_numpy_math() -> None:
    result = check_factor_code(
        "factor = np.log(data['close']).replace([np.inf, -np.inf], np.nan)"
    )

    assert result.passed is True


def test_allows_pandas_close_column_attribute() -> None:
    result = check_factor_code("factor = data.close.pct_change(5).fillna(0.0)")

    assert result.passed is True


def test_allows_safe_pandas_converters() -> None:
    result = check_factor_code("factor = pd.to_numeric(data['close'], errors='coerce')")

    assert result.passed is True
