"""Validate a Windows PyInstaller bundle before offline distribution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

import pefile

COMMON_REQUIRED = (
    "PyQt6/QtCore.pyd",
    "PyQt6/Qt6/bin/Qt6Core.dll",
    "PyQt6/Qt6/bin/Qt6Gui.dll",
    "PyQt6/Qt6/bin/Qt6Widgets.dll",
    "PyQt6/Qt6/plugins/platforms/qwindows.dll",
    "python312.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "anylabeling/services/auto_labeling/osam/clip/"
    "bpe_simple_vocab_16e6.txt.gz",
)

CUDA12_REQUIRED = (
    "onnxruntime/capi/onnxruntime_providers_cuda.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudart64_12.dll",
    "cudnn64_9.dll",
    "cufft64_11.dll",
)

PE_SUFFIXES = {".dll", ".exe", ".pyd"}


def _imported_symbols(path: Path) -> dict[str, set[str]]:
    try:
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
            ]
        )
    except (OSError, pefile.PEFormatError):
        return {}

    imports: dict[str, set[str]] = {}
    for attribute in (
        "DIRECTORY_ENTRY_IMPORT",
        "DIRECTORY_ENTRY_DELAY_IMPORT",
    ):
        for entry in getattr(pe, attribute, ()):
            dll_name = entry.dll.decode("ascii", errors="replace").lower()
            symbols = imports.setdefault(dll_name, set())
            for imported in entry.imports:
                if imported.name:
                    symbols.add(
                        imported.name.decode("ascii", errors="replace")
                    )
    pe.close()
    return imports


def _exported_symbols(path: Path) -> set[str]:
    try:
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]
            ]
        )
    except (OSError, pefile.PEFormatError):
        return set()

    export_directory = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
    if export_directory is None:
        pe.close()
        return set()

    # pefile intentionally caps parsed export names at 8192 to limit malformed
    # files. Qt6Core and numerical runtimes legitimately exceed that count, so
    # walk the PE name table directly after pefile has validated the structure.
    export_struct = export_directory.struct
    symbols = set()
    for index in range(export_struct.NumberOfNames):
        name_rva = pe.get_dword_at_rva(
            export_struct.AddressOfNames + index * 4
        )
        if name_rva is None:
            continue
        name = pe.get_string_at_rva(name_rva)
        if name:
            symbols.add(name.decode("ascii", errors="replace"))
    pe.close()
    return symbols


def _is_windows_component(name: str, system32: Path) -> bool:
    return (
        name.startswith("api-ms-win-")
        or name.startswith("ext-ms-")
        or (system32 / name).is_file()
    )


def verify_bundle(
    bundle: Path, cuda12: bool, onefile_runtime: bool = False
) -> None:
    bundle = bundle.resolve()
    internal = bundle if onefile_runtime else bundle / "_internal"
    if not bundle.is_dir() or not internal.is_dir():
        bundle_type = "onefile runtime" if onefile_runtime else "onedir"
        raise RuntimeError(f"Not a PyInstaller {bundle_type} bundle: {bundle}")

    required = COMMON_REQUIRED + (CUDA12_REQUIRED if cuda12 else ())
    missing = [
        relative
        for relative in required
        if not (internal / relative).is_file()
    ]
    if missing:
        raise RuntimeError("Missing required files: " + ", ".join(missing))

    if list(bundle.rglob("onnxruntime_providers_tensorrt.dll")):
        raise RuntimeError(
            "TensorRT provider is present without the separately licensed "
            "TensorRT runtime"
        )

    files = [path for path in bundle.rglob("*") if path.is_file()]
    bundled_names = {path.name.lower() for path in files}
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    system32 = system_root / "System32"

    bundled_icu = [
        path
        for path in files
        if re.fullmatch(r"icu(?:uc|dt)\d*\.dll", path.name, re.IGNORECASE)
    ]
    if bundled_icu:
        raise RuntimeError(
            "External ICU DLLs would shadow the Windows ICU required by Qt: "
            + ", ".join(str(path.relative_to(bundle)) for path in bundled_icu)
        )

    qt_core = internal / "PyQt6" / "Qt6" / "bin" / "Qt6Core.dll"
    qt_icu_imports = _imported_symbols(qt_core).get("icuuc.dll", set())
    windows_icu = system32 / "icuuc.dll"
    missing_icu_symbols = qt_icu_imports - _exported_symbols(windows_icu)
    if missing_icu_symbols:
        raise RuntimeError(
            "Windows ICU is incompatible with bundled Qt6Core.dll; missing "
            "exports: " + ", ".join(sorted(missing_icu_symbols))
        )

    unresolved: dict[Path, list[str]] = {}
    pe_files = [path for path in files if path.suffix.lower() in PE_SUFFIXES]
    dlls_by_name: dict[str, list[Path]] = {}
    for path in pe_files:
        dlls_by_name.setdefault(path.name.lower(), []).append(path)

    export_cache: dict[Path, set[str]] = {}
    symbol_mismatches: dict[tuple[Path, str], list[str]] = {}
    for path in pe_files:
        imports = _imported_symbols(path)
        missing_imports = sorted(
            name
            for name in imports
            if name not in bundled_names
            and not _is_windows_component(name, system32)
        )
        if missing_imports:
            unresolved[path.relative_to(bundle)] = missing_imports

        for dll_name, imported_names in imports.items():
            candidates = dlls_by_name.get(dll_name, [])
            if not candidates or not imported_names:
                continue

            same_directory = [
                candidate
                for candidate in candidates
                if candidate.parent == path.parent
            ]
            root_candidates = [
                candidate
                for candidate in candidates
                if candidate.parent == internal
            ]
            ordered_candidates = same_directory + root_candidates + candidates
            unique_candidates = list(dict.fromkeys(ordered_candidates))
            candidate_exports = []
            for candidate in unique_candidates:
                if candidate not in export_cache:
                    export_cache[candidate] = _exported_symbols(candidate)
                exports = export_cache[candidate]
                candidate_exports.append(exports)

            if any(
                imported_names.issubset(exports)
                for exports in candidate_exports
            ):
                continue

            available = set().union(*candidate_exports)
            symbol_mismatches[(path.relative_to(bundle), dll_name)] = sorted(
                imported_names - available
            )

    if unresolved:
        details = "; ".join(
            f"{path}: {', '.join(names)}"
            for path, names in sorted(
                unresolved.items(), key=lambda item: str(item[0]).lower()
            )
        )
        raise RuntimeError("Unresolved non-system DLL imports: " + details)

    if symbol_mismatches:
        details = "; ".join(
            f"{path} -> {dll_name}: {', '.join(names[:20])}"
            for (path, dll_name), names in sorted(
                symbol_mismatches.items(),
                key=lambda item: (str(item[0][0]).lower(), item[0][1]),
            )
        )
        raise RuntimeError("Unresolved bundled DLL entry points: " + details)

    print(
        f"PASS: {bundle.name}: {len(files)} files, "
        f"{len(pe_files)} PE files, no unresolved DLL imports or entry points"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--cuda12", action="store_true")
    parser.add_argument(
        "--onefile-runtime",
        action="store_true",
        help="Treat bundle as a running onefile executable's _MEI directory",
    )
    args = parser.parse_args()
    try:
        verify_bundle(args.bundle, args.cuda12, args.onefile_runtime)
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
