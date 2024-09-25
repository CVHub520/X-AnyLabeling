# -*- mode: python -*-
# vim: ft=python

import sys

sys.setrecursionlimit(5000)  # required on Windows

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
        ('C:/Users/18102/.conda/envs/x-anylabeling-gpu/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.dll', 'onnxruntime/capi'),
        ('C:/Users/18102/.conda/envs/x-anylabeling-gpu/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.dll', 'onnxruntime/capi')
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
    name='X-AnyLabeling-GPU',
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