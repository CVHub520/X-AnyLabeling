# -*- mode: python -*-
# vim: ft=python

import sys

a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    datas=[
        ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
        ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
        ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling'),
        ('anylabeling/services/auto_labeling/configs/*.json', 'anylabeling/services/auto_labeling/configs'),
        ('anylabeling/services/auto_labeling/configs/*.txt', 'anylabeling/services/auto_labeling/configs'),
        ('/home/cvhub/miniconda3/envs/x-anylabeling-gpu/lib/python3.8/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so', 'onnxruntime/capi'),
        ('/home/cvhub/miniconda3/envs/x-anylabeling-gpu/lib/python3.8/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so', 'onnxruntime/capi'),
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
