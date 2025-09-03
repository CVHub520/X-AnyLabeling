# -*- mode: python -*-
# vim: ft=python

import sys
import sysconfig
from pathlib import Path

site_packages = Path(sysconfig.get_path("purelib"))
onnxruntime_libs = [
    (str(site_packages / 'onnxruntime/capi/libonnxruntime_providers_cuda.so'), 'onnxruntime/capi'),
    (str(site_packages / 'onnxruntime/capi/libonnxruntime_providers_shared.so'), 'onnxruntime/capi')
]

a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    datas=[
        ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
        ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
        ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling'),
        ('anylabeling/services/auto_labeling/configs/bert/*', 'anylabeling/services/auto_labeling/configs/bert'),
        ('anylabeling/services/auto_labeling/configs/clip/*', 'anylabeling/services/auto_labeling/configs/clip'),
        ('anylabeling/services/auto_labeling/configs/ppocr/*', 'anylabeling/services/auto_labeling/configs/ppocr'),
        ('anylabeling/services/auto_labeling/configs/ram/*', 'anylabeling/services/auto_labeling/configs/ram'),
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
    name='X-Anylabeling-Linux-GPU',
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon='anylabeling/resources/images/icon.icns',
)
app = BUNDLE(
    exe,
    name='X-AnyLabeling.app',
    icon='anylabeling/resources/images/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)