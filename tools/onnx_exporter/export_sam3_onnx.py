"""
Export SAM 3 (ViT-H) to ONNX via samexporter.

This script is a thin wrapper around the samexporter toolkit:
    https://github.com/vietanhdev/samexporter

Three ONNX files are produced (each with an external `.onnx.data` sidecar):
    sam3_image_encoder.onnx      - ViT-H image backbone
    sam3_language_encoder.onnx   - CLIP-based text encoder
    sam3_decoder.onnx            - grounding + mask decoder

Usage:
    1. Clone samexporter and initialise the sam3 submodule:

        git clone https://github.com/vietanhdev/samexporter.git
        cd samexporter
        git submodule update --init sam3

    2. Install dependencies:

        pip install -e ".[export]"
        pip install osam          # required for CLIP tokenisation

    3. Run the export:

        python -m samexporter.export_sam3 \\
            --output_dir output_models/sam3 \\
            --opset 18

       Optionally add --simplify to run onnxsim on the generated files
       (reduces some redundant ops; requires: pip install onnxsim):

        SIMPLIFY=1 bash convert_sam3.sh

    4. Alternatively, run this script directly (it delegates to samexporter):

        python tools/onnx_exporter/export_sam3_onnx.py \\
            --samexporter_dir /path/to/samexporter \\
            --output_dir output_models/sam3 \\
            --opset 18 \\
            [--simplify]

Requirements:
    torch>=2.0, onnx>=1.15, torchvision
    osam (for CLIP tokenisation)
    onnxsim (optional, for --simplify)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM 3 ViT-H to ONNX via samexporter"
    )
    parser.add_argument(
        "--samexporter_dir",
        type=str,
        required=True,
        help="Path to the cloned samexporter repository root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_models/sam3",
        help="Directory to write the exported ONNX files (default: output_models/sam3)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnxsim on the exported models (requires onnxsim)",
    )
    args = parser.parse_args()

    samexporter_dir = Path(args.samexporter_dir).resolve()
    if not samexporter_dir.is_dir():
        print(f"[ERROR] samexporter directory not found: {samexporter_dir}")
        sys.exit(1)

    sam3_submodule = samexporter_dir / "sam3"
    if not sam3_submodule.is_dir():
        print(
            "[ERROR] sam3 submodule not found. "
            "Run: git submodule update --init sam3  inside samexporter."
        )
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "samexporter.export_sam3",
        "--output_dir",
        args.output_dir,
        "--opset",
        str(args.opset),
    ]
    if args.simplify:
        cmd.append("--simplify")

    print(f"[INFO] Running export from: {samexporter_dir}")
    print(f"[INFO] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(samexporter_dir))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
