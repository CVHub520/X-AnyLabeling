# -*- mode: python -*-
# vim: ft=python

import sys
sys.setrecursionlimit(5000)  # required on Windows
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

data_files = collect_data_files('transformers', include_py_files=True, includes=['**/*.py'])
specific_files = [
    ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
    ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
    ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling'),
    ('C:/Users/18102/.conda/envs/x-anylabeling-gpu/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.dll', 'onnxruntime/capi'),
    ('C:/Users/18102/.conda/envs/x-anylabeling-gpu/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.dll', 'onnxruntime/capi')
]

datas = data_files + specific_files
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('numpy')
datas += copy_metadata('pyyaml')
datas += copy_metadata('requests')
datas += copy_metadata('filelock')
datas += copy_metadata('packaging')
datas += copy_metadata('tokenizers')
datas += copy_metadata('safetensors')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('importlib_metadata')


a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    datas=datas,
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
