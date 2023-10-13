# -*- mode: python -*-
# vim: ft=python

import sys
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

data_files = collect_data_files('transformers', include_py_files=True, includes=['**/*.py'])
specific_files = [
    ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
    ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
    ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling'),
    ('/home/cvhub/miniconda3/envs/x-anylabeling-gpu/lib/python3.8/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so', 'onnxruntime/capi'),
    ('/home/cvhub/miniconda3/envs/x-anylabeling-gpu/lib/python3.8/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so', 'onnxruntime/capi'),

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
