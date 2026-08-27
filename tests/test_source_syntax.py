from pathlib import Path
import warnings


def test_package_sources_compile_without_syntax_warnings():
    src_root = Path(__file__).parents[1] / "src" / "jeanspy"
    python_files = sorted(src_root.rglob("*.py"))
    assert python_files

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for path in python_files:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
