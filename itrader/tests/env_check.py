import importlib


def test_core_dependencies_importable():
    required = ["numpy", "pandas", "torch", "ray", "gymnasium", "duckdb"]
    for pkg in required:
        mod = importlib.import_module(pkg)
        assert mod is not None


def test_optional_llm_dependency_importable_if_installed():
    # Optional dependency for proposer backends. This test should not fail if absent.
    try:
        importlib.import_module("llama_cpp")
    except Exception:
        assert True
