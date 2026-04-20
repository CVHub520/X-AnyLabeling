# SAM 3 / EdgeTAM Interactive Video Object Segmentation Example (sam3.cpp backend)

## Introduction

This guide walks you through running **SAM 3** and **EdgeTAM** video tracking inside X-AnyLabeling using the [sam3.cpp](https://github.com/PABannier/sam3.cpp) C++/ggml engine. Compared to the existing PyTorch-backed `segment_anything_2_video` backend, this engine has no `torch` dependency and runs on Apple Metal, NVIDIA CUDA, or plain CPU.

It is **side-by-side** with the original SAM 2 video integration — nothing about that flow changes. The new entries appear in the AI model picker as `sam3cpp_video-*` and download their `.ggml` weights automatically from [huggingface.co/PABannier/sam3.cpp](https://huggingface.co/PABannier/sam3.cpp) on first use.

## Installation

**Step 0:** Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), the fast Python package and project manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# Windows (PowerShell):
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Step 1:** Create a Python `3.10`–`3.13` virtual environment and activate it:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate          # bash / zsh
# .venv\Scripts\Activate.ps1       # Windows PowerShell
```

**Step 2:** Build sam3.cpp from source. The instructions below come from the upstream [sam3.cpp Quick Start](https://github.com/PABannier/sam3.cpp#quick-start) and [Building from Source](https://github.com/PABannier/sam3.cpp#building-from-source) sections. Prerequisites (also from upstream): a C++14 compiler (Clang, GCC, or MSVC) and CMake 3.14+.

```bash
git clone --recursive https://github.com/PABannier/sam3.cpp
cd sam3.cpp
mkdir build && cd build
cmake ..
make -j
```

Metal GPU is enabled automatically on macOS. To disable it, pass `-DSAM3_METAL=OFF`. To enable NVIDIA CUDA on Linux/Windows, pass `-DGGML_CUDA=ON` (CUDA Toolkit 11.4+ required).

On Windows, replace `make -j` with `cmake --build . --config Release -j` from a Developer PowerShell for VS 2022.

**Step 3:** Build the `sam3cpp` Python bindings. The upstream sam3.cpp library is C++-only; the Python wrapper this integration consumes is being upstreamed in a separate PR (link will be added to this document once the PR is open). Until that PR merges, follow the maintainer notes in the X-AnyLabeling PR description for the manual binding build.

Once the upstream PR is merged, this step becomes a single CMake flag added to Step 2:

```bash
cmake .. -DSAM3_BUILD_PYBIND=ON
make -j
```

**Step 4:** Make the `sam3cpp` module importable from your virtual environment. Either keep `PYTHONPATH` exported in the shell that launches X-AnyLabeling, or copy the built extension into your environment's `site-packages/`:

```bash
# Per-shell (bash/zsh):
export PYTHONPATH="/path/to/sam3.cpp/build:$PYTHONPATH"

# Permanent: copy the extension into the active environment
cp /path/to/sam3.cpp/build/sam3cpp.* "$(python -c 'import site; print(site.getsitepackages()[0])')"
```

Verify the import:

```bash
python -c "import sam3cpp; print(sam3cpp.__doc__)"
```

**Step 5:** Install X-AnyLabeling itself. Clone the repo and follow the standard installation guide ([简体中文](../../../docs/zh_cn/get_started.md) | [English](../../../docs/en/get_started.md)):

```bash
git clone https://github.com/CVHub520/X-AnyLabeling
cd X-AnyLabeling
```

If `sam3cpp` is not importable, X-AnyLabeling reports a clean error message in the model picker pointing at this document — the rest of the application keeps working.

## Getting Started

**Step 0:** Launch the app:

```bash
python3 anylabeling/app.py
```

**Step 1:** Load a sam3.cpp video model from the AI model picker. Available entries:

| Model | Size | Best for |
|---|---|---|
| `sam3cpp_video-edgetam-q4_0` | 15 MB | mobile / lowest latency |
| `sam3cpp_video-edgetam-f16` | 27 MB | best EdgeTAM quality |
| `sam3cpp_video-sam3_visual-q4_0` | 289 MB | SAM 3 visual tracking, quantized |
| `sam3cpp_video-sam3_visual-f16` | 946 MB | SAM 3 visual tracking, full precision |
| `sam3cpp_video-sam3-q4_0` | 707 MB | SAM 3 full (text + visual), quantized |

Weights download automatically on first use into `~/xanylabeling_data/models/<model_name>/`.

**Step 2:** Add a video file (`Ctrl + O`) or a folder of split video frames (`Ctrl + U`). Supported file formats are limited to `[*.jpg, *.jpeg, *.JPG, *.JPEG]`; video files are auto-converted to JPEG.

## Usage

The interaction model is identical to the existing SAM 2 video example — see [the SAM 2 video tutorial](../sam2/README.md#usage) for the full walkthrough of:

- Adding positive / negative point prompts and rectangles on the first frame
- Confirming an object with `Finish (f)` and assigning labels + track IDs
- Propagating prompts across the video (`Ctrl+M` or the play button)
- Refining the tracklet by pausing, resetting the tracker, and re-prompting
- Renaming labels and editing track IDs after tracking finishes

All shortcuts and buttons work identically on the sam3.cpp backend.

## Relationship to `segment_anything_2_video`

The PyTorch-backed `segment_anything_2_video` model continues to ship and remains the right choice for users who already have a working PyTorch + CUDA environment with the `sam-2` Python package installed. The `sam3cpp_video` engine is additive: it brings EdgeTAM and SAM 3 to X-AnyLabeling for the first time and gives Apple Silicon (and CPU-only) users a competitive video-tracking path without a PyTorch install.

---

Congratulations! 🎉 You're set up to use the sam3.cpp engine from X-AnyLabeling. Open an issue if you hit any problems.
