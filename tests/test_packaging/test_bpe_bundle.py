from pathlib import Path

from anylabeling.services.auto_labeling.osam.clip.simple_tokenizer import (
    SimpleTokenizer,
    default_bpe,
)

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DESTINATION = "anylabeling/services/auto_labeling/osam/clip"


def test_bpe_vocabulary_is_present_at_tokenizer_runtime_path():
    vocabulary = Path(default_bpe())

    assert vocabulary.is_file()
    assert vocabulary.name == "bpe_simple_vocab_16e6.txt.gz"
    assert vocabulary.stat().st_size == 1_356_917
    assert SimpleTokenizer().encode("pixel edge")


def test_windows_cpu_and_gpu_specs_bundle_bpe_at_runtime_path():
    for name in ("x-anylabeling-win-cpu.spec", "x-anylabeling-win-gpu.spec"):
        content = (
            ROOT / "packaging" / "pyinstaller" / "specs" / name
        ).read_text(encoding="utf-8")

        assert "bpe_simple_vocab_16e6.txt.gz" in content
        assert RESOURCE_DESTINATION in content


def test_windows_specs_bootstrap_and_require_bundled_qt_runtime():
    for name in ("x-anylabeling-win-cpu.spec", "x-anylabeling-win-gpu.spec"):
        content = (
            ROOT / "packaging" / "pyinstaller" / "specs" / name
        ).read_text(encoding="utf-8")

        assert "qt_dll_bootstrap.py" in content
        assert "required_qt_dlls" in content
        assert "qt6core.dll" in content
        assert "qwindows.dll" in content
        assert "_strip_external_icu_binaries" in content


def test_qt_bootstrap_runs_before_application_imports():
    hook = (
        ROOT
        / "packaging"
        / "pyinstaller"
        / "runtime_hooks"
        / "qt_dll_bootstrap.py"
    ).read_text(encoding="utf-8")

    assert "sys._MEIPASS" in hook
    assert '"PyQt6", "Qt6", "bin"' in hook
    assert "os.add_dll_directory" in hook
    assert "ctypes.WinDLL" in hook
    assert "_SAFE_LOAD_FLAGS" in hook


def test_gpu_spec_explicitly_bundles_cuda12_runtime():
    content = (
        ROOT
        / "packaging"
        / "pyinstaller"
        / "specs"
        / "x-anylabeling-win-gpu.spec"
    ).read_text(encoding="utf-8")

    assert "_collect_cuda12_runtime_dlls" in content
    for dll_name in (
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudart64_12.dll",
        "cudnn64_9.dll",
        "cufft64_11.dll",
    ):
        assert dll_name in content
