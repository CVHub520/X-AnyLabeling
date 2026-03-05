# -*- mode: python -*-
# vim: ft=python

import glob
import os
import re
import sys
import sysconfig
from pathlib import Path

def _resolve_root_dir():
    env_root = os.environ.get('X_ANYLABELING_ROOT')
    if env_root:
        return os.path.abspath(env_root)

    spec_path = globals().get('SPEC')
    if isinstance(spec_path, str) and spec_path:
        return os.path.abspath(os.path.join(os.path.dirname(spec_path), '..', '..', '..'))

    return os.path.abspath(os.getcwd())

ROOT_DIR = _resolve_root_dir()

def _p(*parts):
    return os.path.join(ROOT_DIR, *parts)

def _load_version():
    app_info_path = _p('anylabeling', 'app_info.py')
    with open(app_info_path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Failed to read __version__ from: {app_info_path}")
    return match.group(1)

__version__ = _load_version()

site_packages = Path(sysconfig.get_path("purelib"))
onnxruntime_libs = [
    (str(site_packages / 'onnxruntime/capi/libonnxruntime_providers_cuda.so'), 'onnxruntime/capi'),
    (str(site_packages / 'onnxruntime/capi/libonnxruntime_providers_shared.so'), 'onnxruntime/capi')
]

_lib_dir = sysconfig.get_config_var("LIBDIR") or ""
_expat_libs = [
    f for f in glob.glob(os.path.join(_lib_dir, "libexpat.so*"))
    if os.path.isfile(f)
]
expat_binaries = [(_lib, ".") for _lib in _expat_libs]

a = Analysis(
    [_p('anylabeling', 'app.py')],
    pathex=[_p('anylabeling')],
    binaries=expat_binaries,
    datas=[
        (_p('anylabeling', 'configs', 'auto_labeling', '*.yaml'), 'anylabeling/configs/auto_labeling'),
        (_p('anylabeling', 'configs', '*.yaml'), 'anylabeling/configs'),
        (_p('anylabeling', 'views', 'labeling', 'widgets', 'auto_labeling', 'auto_labeling.ui'), 'anylabeling/views/labeling/widgets/auto_labeling'),
        (_p('anylabeling', 'services', 'auto_labeling', 'configs', 'bert', '*'), 'anylabeling/services/auto_labeling/configs/bert'),
        (_p('anylabeling', 'services', 'auto_labeling', 'configs', 'clip', '*'), 'anylabeling/services/auto_labeling/configs/clip'),
        (_p('anylabeling', 'services', 'auto_labeling', 'configs', 'ppocr', '*'), 'anylabeling/services/auto_labeling/configs/ppocr'),
        (_p('anylabeling', 'services', 'auto_labeling', 'configs', 'ram', '*'), 'anylabeling/services/auto_labeling/configs/ram'),
        *onnxruntime_libs
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name=f'X-AnyLabeling-v{__version__}-Linux-GPU',
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon=_p('anylabeling', 'resources', 'images', 'icon.icns'),
)
app = BUNDLE(
    exe,
    name='X-AnyLabeling.app',
    icon=_p('anylabeling', 'resources', 'images', 'icon.icns'),
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
