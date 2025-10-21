import importlib
import pytest

@pytest.mark.parametrize("pkg", ["numba", "joblib"])
def test_optional_perf_extras(pkg):
    mod = importlib.util.find_spec(pkg)
    # If not present, that's fine; if present, it should import.
    if mod is not None:
        importlib.import_module(pkg)
