PS C:\Users\manuel.campos\OneDrive - RUSSULA S.A\Escritorio\CLS> pytest tests/test_integration.py -v
================================================= test session starts =================================================
platform win32 -- Python 3.12.6, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\manuel.campos\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\manuel.campos\OneDrive - RUSSULA S.A\Escritorio\CLS
collected 1 item

tests/test_integration.py::test_full_pipeline PASSED                                                             [100%]

================================================== warnings summary ===================================================
tests/test_integration.py::test_full_pipeline
tests/test_integration.py::test_full_pipeline
  C:\Users\manuel.campos\AppData\Local\Programs\Python\Python312\Lib\site-packages\seaborn\categorical.py:700: PendingDeprecationWarning: vert: bool will be deprecated in a future version. Use orientation: {'vertical', 'horizontal'} instead.
    artists = ax.bxp(**boxplot_kws)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================ 1 passed, 2 warnings in 7.62s ============================================