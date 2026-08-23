# Vision Prompt environment report

Validated on 2026-08-24 (Asia/Shanghai).

## Source baseline

| Component | Value |
| --- | --- |
| X-AnyLabeling | 4.0.3 (`upstream/main`) |
| X-AnyLabeling commit | `8eb3fdcf648ce3302e1d13bfbc605d645a57b5ec` |
| Working branch | `feature/yoloe-cross-image-vpe-batch` |
| Upstream | `https://github.com/CVHub520/X-AnyLabeling.git` |
| THU-MIG/yoloe commit | `40cd606cabdbe2b566d6f14a6b162c89206e9a1b` |

The project is an independent repository rooted at `VisionPromptStudio`.
An unrelated Git repository in the user home directory is not used.

## Runtime

| Component | Value |
| --- | --- |
| Python | 3.11.9, isolated in `.venv` |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime in PyTorch | 13.0 |
| CUDA available | Yes |
| cuDNN | 9.2.0 (`92000`) |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB |
| Compute capability | 12.0 |
| NVIDIA driver | 596.49 |
| CUDA Toolkit / `nvcc` | Not installed; not required by the wheel runtime |
| Ultralytics | 8.3.39 from the checked-out THU-MIG/yoloe source |
| PyQt6 | 6.11.0 |
| supervision | 0.30.0 |
| NumPy | 2.4.6 |

The upstream `pyproject.toml` now requires Python 3.11 or newer and declares
3.11-3.13 support. The older YOLOE example still mentioning Python 3.10 is
therefore not used as the environment authority for this baseline.

## Models

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `yoloe-11s-seg.pt` | 27,803,986 | `8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3` |
| `yoloe-11s-seg-pf.pt` | 24,239,013 | `22ed131030aaf1985d34f2ed58576c2fb7f30d5dbe0c435574a1c073c475920f` |
| `mobileclip_blt.pt` | 599,214,572 | `670844f7a886dd6eff7a9285adfc53f3d3c889c03bfc8354010cb5c6bf27441a` |

The model files live under ignored `.cache/models/` and are not committed.
YOLOE weights came from the official `jameslahm/yoloe` Hugging Face repository;
MobileCLIP came from the official `apple/MobileCLIP-B-LT` repository. Their
byte sizes match the URLs configured or documented by upstream.

## Baseline result

- X-AnyLabeling 4.0.3 Qt event loop and main window initialized successfully
  with the offscreen Qt platform.
- GPU tensor allocation and computation passed on `cuda:0`.
- X-AnyLabeling YOLOE Text Prompt passed: `person.car` produced four editable
  rectangle Shapes on `bus.jpg`.
- X-AnyLabeling intra-image Visual Prompt passed: one reference rectangle
  produced three editable rectangle Shapes on `bus.jpg`.
- `pip check`: no broken requirements.
