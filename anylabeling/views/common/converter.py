import os
import os.path as osp
import sys
from glob import glob
from pathlib import Path
from termcolor import colored
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.logger import logger


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
LABEL_EXTENSIONS = {
    "dota": ".txt",
    "mask": ".png",
    "voc": ".xml",
    "xlabel": ".json",
    "yolo": ".txt",
}

COCO_TASK_MODES = ["detect", "segment", "pose"]
PPOCR_TASK_MODES = ["rec", "kie"]
VOC_TASK_MODES = ["detect", "segment"]
YOLO_TASK_MODES = ["detect", "segment", "obb", "pose"]

SUPPORTED_TASKS = {
    # Import to XLABEL
    "yolo2xlabel": {
        "description": "Convert YOLO format to XLABEL",
        "modes": ["detect", "segment", "obb", "pose"],
        "required_args": ["images", "labels", "output"],
        "conditional_args": {
            "detect": ["classes"],
            "segment": ["classes"],
            "obb": ["classes"],
            "pose": ["pose_cfg"],
        },
    },
    "voc2xlabel": {
        "description": "Convert VOC format to XLABEL",
        "modes": ["detect", "segment"],
        "required_args": ["labels", "output"],
        "conditional_args": {},
    },
    "coco2xlabel": {
        "description": "Convert COCO format to XLABEL",
        "modes": ["detect", "segment", "pose"],
        "required_args": ["labels", "output"],
        "conditional_args": {
            "detect": ["classes"],
            "segment": ["classes"],
            "pose": ["pose_cfg"],
        },
    },
    "dota2xlabel": {
        "description": "Convert DOTA format to XLABEL",
        "modes": [],
        "required_args": ["images", "output"],
        "conditional_args": {},
    },
    "mot2xlabel": {
        "description": "Convert MOT format to XLABEL",
        "modes": [],
        "required_args": ["labels", "images", "output", "classes"],
        "conditional_args": {},
    },
    "ppocr2xlabel": {
        "description": "Convert PaddleOCR format to XLABEL",
        "modes": ["rec", "kie"],
        "required_args": ["labels", "images", "output", "mode"],
        "conditional_args": {},
    },
    "mask2xlabel": {
        "description": "Convert mask format to XLABEL",
        "modes": [],
        "required_args": ["images", "output", "mapping"],
        "conditional_args": {},
    },
    "vlmr12xlabel": {
        "description": "Convert VLM-R1 format to XLABEL",
        "modes": [],
        "required_args": ["images", "output"],
        "conditional_args": {},
    },
    "odvg2xlabel": {
        "description": "Convert ODVG format to XLABEL",
        "modes": [],
        "required_args": ["labels", "output"],
        "conditional_args": {},
    },
    # Export from XLABEL
    "xlabel2yolo": {
        "description": "Convert XLABEL to YOLO format",
        "modes": ["detect", "segment", "obb", "pose"],
        "required_args": ["images", "output", "mode"],
        "conditional_args": {
            "detect": ["classes"],
            "segment": ["classes"],
            "obb": ["classes"],
            "pose": ["pose_cfg"],
        },
        "optional_args": ["skip_empty_files"],
    },
    "xlabel2voc": {
        "description": "Convert XLABEL to VOC format",
        "modes": ["detect", "segment"],
        "required_args": ["images", "output", "mode"],
        "conditional_args": {},
        "optional_args": ["skip_empty_files"],
    },
    "xlabel2coco": {
        "description": "Convert XLABEL to COCO format",
        "modes": ["detect", "segment", "pose"],
        "required_args": ["images", "output", "mode"],
        "conditional_args": {
            "detect": ["classes"],
            "segment": ["classes"],
            "pose": ["pose_cfg"],
        },
    },
    "xlabel2dota": {
        "description": "Convert XLABEL to DOTA format",
        "modes": [],
        "required_args": ["images", "output"],
        "conditional_args": {},
    },
    "xlabel2mask": {
        "description": "Convert XLABEL to mask format",
        "modes": [],
        "required_args": ["images", "output", "mapping"],
        "conditional_args": {},
    },
    "xlabel2mot": {
        "description": "Convert XLABEL to MOT format",
        "modes": [],
        "required_args": ["labels", "output", "classes"],
        "conditional_args": {},
    },
    "xlabel2mots": {
        "description": "Convert XLABEL to MOTS format",
        "modes": [],
        "required_args": ["labels", "output", "classes"],
        "conditional_args": {},
    },
    "xlabel2odvg": {
        "description": "Convert XLABEL to ODVG format",
        "modes": [],
        "required_args": ["images", "output", "classes"],
        "conditional_args": {},
    },
    "xlabel2vlmr1": {
        "description": "Convert XLABEL to VLM-R1 format",
        "modes": [],
        "required_args": ["images", "output"],
        "conditional_args": {},
    },
    "xlabel2ppocr": {
        "description": "Convert XLABEL to PaddleOCR format",
        "modes": ["rec", "kie"],
        "required_args": ["images", "output", "mode"],
        "conditional_args": {},
    },
}


def get_image_files(image_dir):
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob(osp.join(image_dir, f"*{ext}")))
        image_files.extend(glob(osp.join(image_dir, f"*{ext.upper()}")))
    return sorted(image_files)


def get_label_files(label_dir, label_ext):
    """Get all label files with the specified extension"""
    label_files = glob(osp.join(label_dir, f"*{label_ext}"))
    return sorted(label_files)


def find_matching_file(image_path, label_dir, label_ext):
    image_name = osp.splitext(osp.basename(image_path))[0]
    label_file = osp.join(label_dir, f"{image_name}{label_ext}")
    return label_file if osp.exists(label_file) else None


def list_supported_tasks():
    """List all supported conversion tasks"""
    print(colored("\n" + "=" * 80, "cyan"))
    print(colored("SUPPORTED CONVERSION TASKS", "cyan", attrs=["bold"]))
    print(colored("=" * 80 + "\n", "cyan"))

    import_tasks = {k: v for k, v in SUPPORTED_TASKS.items() if "2xlabel" in k}
    export_tasks = {k: v for k, v in SUPPORTED_TASKS.items() if "xlabel2" in k}

    print(colored("📥 IMPORT TO XLABEL", "green", attrs=["bold"]))
    print(colored("-" * 80, "green"))
    for task_name, task_info in import_tasks.items():
        modes_str = (
            f" [{', '.join(task_info['modes'])}]" if task_info["modes"] else ""
        )
        print(f"  • {colored(task_name, 'yellow')}{modes_str}")

    print(colored("\n📤 EXPORT FROM XLABEL", "blue", attrs=["bold"]))
    print(colored("-" * 80, "blue"))
    for task_name, task_info in export_tasks.items():
        modes_str = (
            f" [{', '.join(task_info['modes'])}]" if task_info["modes"] else ""
        )
        print(f"  • {colored(task_name, 'yellow')}{modes_str}")

    print(colored("\n" + "=" * 80, "cyan"))
    print(colored(f"Total: {len(SUPPORTED_TASKS)} conversion tasks", "cyan"))
    print(colored("=" * 80 + "\n", "cyan"))

    print(colored("Usage:", "white", attrs=["bold"]))
    print("  xanylabeling convert                          # Show all tasks")
    print(
        "  xanylabeling convert --task <task>            # Show detailed help for a task"
    )
    print("  xanylabeling convert --task <task> [options]  # Run conversion")
    print()


def show_task_help(task_name):
    """Show detailed help for a specific task with examples"""
    if task_name not in SUPPORTED_TASKS:
        print(colored(f"\n✗ Unknown task: '{task_name}'", "red"))
        print(
            colored(
                "Use 'xanylabeling convert' to see all available tasks.\n",
                "yellow",
            )
        )
        return

    task_info = SUPPORTED_TASKS[task_name]

    print(colored("\n" + "=" * 80, "cyan"))
    print(colored(f"TASK: {task_name}", "cyan", attrs=["bold"]))
    print(colored("=" * 80 + "\n", "cyan"))

    print(colored("Description:", "white", attrs=["bold"]))
    print(f"  {task_info['description']}\n")

    if task_info["modes"]:
        print(colored("Modes:", "white", attrs=["bold"]))
        print(f"  {', '.join(task_info['modes'])}\n")

    print(colored("Required Arguments:", "white", attrs=["bold"]))
    for arg in task_info["required_args"]:
        print(f"  --{arg}")
    print()

    if task_info["conditional_args"]:
        print(colored("Mode-Specific Arguments:", "white", attrs=["bold"]))
        for mode, args in task_info["conditional_args"].items():
            print(f"  {mode}: {', '.join(['--' + arg for arg in args])}")
        print()

    if task_info.get("optional_args"):
        print(colored("Optional Arguments:", "white", attrs=["bold"]))
        for arg in task_info["optional_args"]:
            print(f"  --{arg}")
        print()

    # Show examples based on task type
    print(colored("Examples:", "green", attrs=["bold"]))

    if task_name == "yolo2xlabel":
        print(f"  # Detection")
        print(
            f"  xanylabeling convert --task yolo2xlabel --mode detect --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --classes classes.txt\n")
        print(f"  # Segmentation")
        print(
            f"  xanylabeling convert --task yolo2xlabel --mode segment --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --classes classes.txt\n")
        print(f"  # OBB (Oriented Bounding Box)")
        print(
            f"  xanylabeling convert --task yolo2xlabel --mode obb --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --classes classes.txt\n")
        print(f"  # Pose")
        print(
            f"  xanylabeling convert --task yolo2xlabel --mode pose --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --pose-cfg pose_config.yaml\n")

    elif task_name == "xlabel2yolo":
        print(f"  # Detection")
        print(
            f"  xanylabeling convert --task xlabel2yolo --mode detect --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --classes classes.txt\n")
        print(f"  # Segmentation (skip empty files)")
        print(
            f"  xanylabeling convert --task xlabel2yolo --mode segment --images ./images --labels ./labels \\"
        )
        print(
            f"    --output ./output --classes classes.txt --skip-empty-files\n"
        )
        print(f"  # OBB")
        print(
            f"  xanylabeling convert --task xlabel2yolo --mode obb --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --classes classes.txt\n")
        print(f"  # Pose")
        print(
            f"  xanylabeling convert --task xlabel2yolo --mode pose --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --pose-cfg pose_config.yaml\n")

    elif task_name == "voc2xlabel":
        print(f"  # Detection")
        print(
            f"  xanylabeling convert --task voc2xlabel --mode detect --labels ./Annotations --output ./output\n"
        )
        print(f"  # Segmentation")
        print(
            f"  xanylabeling convert --task voc2xlabel --mode segment --labels ./Annotations --output ./output\n"
        )

    elif task_name == "xlabel2voc":
        print(f"  # Detection")
        print(
            f"  xanylabeling convert --task xlabel2voc --mode detect --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output\n")
        print(f"  # Segmentation (skip empty files)")
        print(
            f"  xanylabeling convert --task xlabel2voc --mode segment --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --skip-empty-files\n")

    elif task_name == "coco2xlabel":
        print(f"  # Detection")
        print(
            f"  xanylabeling convert --task coco2xlabel --mode detect --labels annotations.json \\"
        )
        print(f"    --output ./output --classes classes.txt\n")
        print(f"  # Segmentation")
        print(
            f"  xanylabeling convert --task coco2xlabel --mode segment --labels annotations.json \\"
        )
        print(f"    --output ./output --classes classes.txt\n")
        print(f"  # Pose")
        print(
            f"  xanylabeling convert --task coco2xlabel --mode pose --labels annotations.json \\"
        )
        print(f"    --output ./output --pose-cfg pose_config.yaml\n")

    elif task_name == "xlabel2coco":
        print(f"  # Detection")
        print(
            f"  xanylabeling convert --task xlabel2coco --mode detect --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --classes classes.txt\n")
        print(f"  # Segmentation")
        print(
            f"  xanylabeling convert --task xlabel2coco --mode segment --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --classes classes.txt\n")
        print(f"  # Pose")
        print(
            f"  xanylabeling convert --task xlabel2coco --mode pose --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --pose-cfg pose_config.yaml\n")

    elif task_name == "dota2xlabel":
        print(
            f"  xanylabeling convert --task dota2xlabel --images ./images --labels ./labels --output ./output\n"
        )

    elif task_name == "xlabel2dota":
        print(
            f"  xanylabeling convert --task xlabel2dota --images ./images --labels ./labels --output ./output\n"
        )

    elif task_name == "mask2xlabel":
        print(
            f"  xanylabeling convert --task mask2xlabel --images ./images --labels ./masks \\"
        )
        print(f"    --output ./output --mapping mapping.json\n")

    elif task_name == "xlabel2mask":
        print(
            f"  xanylabeling convert --task xlabel2mask --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --mapping mapping.json\n")

    elif task_name == "mot2xlabel":
        print(
            f"  xanylabeling convert --task mot2xlabel --labels gt.txt --images ./frames \\"
        )
        print(f"    --output ./output --classes classes.txt\n")

    elif task_name == "xlabel2mot":
        print(
            f"  xanylabeling convert --task xlabel2mot --labels ./labels --output ./output --classes classes.txt\n"
        )

    elif task_name == "xlabel2mots":
        print(
            f"  xanylabeling convert --task xlabel2mots --labels ./labels --output ./output --classes classes.txt\n"
        )

    elif task_name == "ppocr2xlabel":
        print(f"  # Recognition")
        print(
            f"  xanylabeling convert --task ppocr2xlabel --labels Label.txt --images ./images \\"
        )
        print(f"    --output ./output --mode rec\n")
        print(f"  # Key Information Extraction")
        print(
            f"  xanylabeling convert --task ppocr2xlabel --labels Label.txt --images ./images \\"
        )
        print(f"    --output ./output --mode kie\n")

    elif task_name == "xlabel2ppocr":
        print(f"  # Recognition")
        print(
            f"  xanylabeling convert --task xlabel2ppocr --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --mode rec\n")
        print(f"  # Key Information Extraction")
        print(
            f"  xanylabeling convert --task xlabel2ppocr --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --mode kie\n")

    elif task_name == "odvg2xlabel":
        print(
            f"  xanylabeling convert --task odvg2xlabel --labels annotations.jsonl --output ./output\n"
        )

    elif task_name == "xlabel2odvg":
        print(
            f"  xanylabeling convert --task xlabel2odvg --images ./images --labels ./labels \\"
        )
        print(f"    --output ./output --classes classes.txt\n")

    elif task_name == "vlmr12xlabel":
        print(
            f"  xanylabeling convert --task vlmr12xlabel --images ./images --labels ./labels --output ./output\n"
        )

    elif task_name == "xlabel2vlmr1":
        print(
            f"  xanylabeling convert --task xlabel2vlmr1 --images ./images --labels ./labels \\"
        )
        print(f"    --output output.jsonl\n")

    print(colored("=" * 80 + "\n", "cyan"))


def validate_task(task_name):
    """Validate if a task is supported"""
    if task_name not in SUPPORTED_TASKS:
        available_tasks = ", ".join(SUPPORTED_TASKS.keys())
        raise ValueError(
            f"Unknown task: '{task_name}'\n"
            f"Available tasks: {available_tasks}\n"
            f"Use 'xanylabeling convert' to see all supported tasks."
        )
    return True


def handle_convert_command(args):
    """Handle the convert command from argparse"""
    if not args.task:
        list_supported_tasks()
        return

    task = args.task.lower()
    validate_task(task)

    provided_args = [
        args.images,
        args.labels,
        args.output,
        args.classes,
        args.pose_cfg,
        args.mode,
        args.mapping,
    ]

    if all(arg is None for arg in provided_args):
        show_task_help(task)
        return

    run_conversion(
        task,
        images=args.images,
        labels=args.labels,
        output=args.output,
        classes_file=args.classes,
        pose_cfg_file=args.pose_cfg,
        mode=args.mode,
        mapping_file=args.mapping,
        skip_empty_files=args.skip_empty_files,
    )


def run_conversion(
    task,
    images=None,
    labels=None,
    output=None,
    classes_file=None,
    pose_cfg_file=None,
    mode=None,
    mapping_file=None,
    skip_empty_files=False,
):
    """Core conversion logic"""
    dota_ext = LABEL_EXTENSIONS["dota"]
    mask_ext = LABEL_EXTENSIONS["mask"]
    yolo_ext = LABEL_EXTENSIONS["yolo"]
    voc_ext = LABEL_EXTENSIONS["voc"]
    xlabel_ext = LABEL_EXTENSIONS["xlabel"]

    try:
        converter = LabelConverter(
            classes_file=classes_file,
            pose_cfg_file=pose_cfg_file,
        )

        if "2xlabel" in task:

            if task == "yolo2xlabel":
                if not images:
                    raise ValueError(
                        f"--images is required for yolo {mode} conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not mode:
                    raise ValueError("--mode is required for YOLO conversion")
                if mode not in YOLO_TASK_MODES:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be one of {YOLO_TASK_MODES}"
                    )
                if mode in ["detect", "segment", "obb"] and not classes_file:
                    raise ValueError(
                        f"--classes is required for yolo {mode} conversion"
                    )
                elif mode == "pose" and not pose_cfg_file:
                    raise ValueError(
                        "--pose-cfg is required for yolo pose conversion"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not output:
                    output = images
                    logger.warning(
                        f"--output not specified, using image directory: {output}"
                    )
                os.makedirs(output, exist_ok=True)

                count = 0
                image_files = get_image_files(images)
                for image_file in tqdm(
                    image_files, desc=f"Converting YOLO {mode} to XLABEL"
                ):
                    label_file = find_matching_file(
                        image_file, labels, yolo_ext
                    )
                    if not label_file:
                        logger.warning(
                            f"Label file not found for: {osp.basename(image_file)}"
                        )
                        continue

                    output_file = osp.join(
                        output,
                        osp.splitext(osp.basename(image_file))[0] + xlabel_ext,
                    )

                    if mode == "detect":
                        converter.yolo_to_custom(
                            label_file, output_file, image_file, "hbb"
                        )
                    elif mode == "segment":
                        converter.yolo_to_custom(
                            label_file, output_file, image_file, "seg"
                        )
                    elif mode == "obb":
                        converter.yolo_obb_to_custom(
                            label_file, output_file, image_file
                        )
                    elif mode == "pose":
                        converter.yolo_pose_to_custom(
                            label_file, output_file, image_file
                        )
                    count += 1

                print(
                    colored(
                        f"✓ Converted {count} files to XLABEL format: {output}",
                        "green",
                    )
                )

            elif task == "voc2xlabel":
                if not mode:
                    raise ValueError("--mode is required for VOC conversion")
                if mode not in VOC_TASK_MODES:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be one of {VOC_TASK_MODES}"
                    )

                if not labels:
                    raise ValueError("--labels is required for VOC conversion")

                if not output:
                    raise ValueError("--output is required for VOC conversion")
                os.makedirs(output, exist_ok=True)

                count = 0
                label_files = get_label_files(labels, voc_ext)
                if not label_files:
                    raise FileNotFoundError(
                        f"No VOC XML files found in: {labels}"
                    )

                for label_file in tqdm(
                    label_files, desc=f"Converting VOC {mode} to XLABEL"
                ):
                    base_name = osp.splitext(osp.basename(label_file))[0]
                    output_file = osp.join(output, base_name + xlabel_ext)

                    if mode == "detect":
                        converter.voc_to_custom(
                            label_file, output_file, "", "rectangle"
                        )
                    elif mode == "segment":
                        converter.voc_to_custom(
                            label_file, output_file, "", "polygon"
                        )
                    count += 1

                print(
                    colored(
                        f"✓ Converted {count} files to XLABEL format: {output}",
                        "green",
                    )
                )

            elif task == "coco2xlabel":
                if not mode:
                    raise ValueError("--mode is required for COCO conversion")
                if mode not in COCO_TASK_MODES:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be one of {COCO_TASK_MODES}"
                    )
                if mode in ["detect", "segment"] and not classes_file:
                    raise ValueError(
                        f"--classes is required for coco {mode} conversion"
                    )
                elif mode == "pose" and not pose_cfg_file:
                    raise ValueError(
                        "--pose-cfg is required for coco pose conversion"
                    )

                if not labels:
                    raise ValueError(
                        "--labels is required (path to COCO JSON file)"
                    )
                if not osp.exists(labels):
                    raise FileNotFoundError(
                        f"COCO JSON file not found: {labels}"
                    )
                if not labels.endswith(".json"):
                    raise ValueError(
                        f"COCO label file must be a JSON file: {labels}"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for COCO conversion"
                    )
                os.makedirs(output, exist_ok=True)

                coco_mode_map = {
                    "detect": "rectangle",
                    "segment": "polygon",
                    "pose": "pose",
                }
                coco_mode = coco_mode_map[mode]

                print(
                    colored(
                        f"Converting COCO {mode} format to XLABEL...", "cyan"
                    )
                )
                converter.coco_to_custom(labels, output, coco_mode)

                output_files = glob(osp.join(output, "*.json"))
                print(
                    colored(
                        f"✓ Converted COCO to {len(output_files)} XLABEL files: {output}",
                        "green",
                    )
                )

            elif task == "dota2xlabel":
                if not images:
                    raise ValueError(
                        "--images is required for DOTA conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for DOTA conversion"
                    )
                os.makedirs(output, exist_ok=True)

                count = 0
                image_files = get_image_files(images)
                for image_file in tqdm(
                    image_files, desc="Converting DOTA to XLABEL"
                ):
                    label_file = find_matching_file(
                        image_file, labels, dota_ext
                    )
                    if not label_file:
                        logger.warning(
                            f"Label file not found for: {osp.basename(image_file)}"
                        )
                        continue

                    output_file = osp.join(
                        output,
                        osp.splitext(osp.basename(image_file))[0] + xlabel_ext,
                    )

                    converter.dota_to_custom(
                        label_file, output_file, image_file
                    )
                    count += 1

                print(
                    colored(
                        f"✓ Converted {count} files to XLABEL format: {output}",
                        "green",
                    )
                )

            elif task == "mask2xlabel":
                if not images:
                    raise ValueError(
                        "--images is required for mask conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not mapping_file:
                    raise ValueError(
                        "--mapping is required (path to mapping table file)"
                    )
                if not osp.exists(mapping_file):
                    raise FileNotFoundError(
                        f"Mapping file not found: {mapping_file}"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for mask conversion"
                    )
                os.makedirs(output, exist_ok=True)

                count = 0
                image_files = get_image_files(images)
                mapping_table = converter.read_json(mapping_file)
                for image_file in tqdm(
                    image_files, desc="Converting mask to XLABEL"
                ):
                    base_name = osp.splitext(osp.basename(image_file))[0]
                    mask_file = osp.join(labels, base_name + mask_ext)
                    if not osp.exists(mask_file):
                        mask_file = osp.join(labels, base_name + ".jpg")
                    if not osp.exists(mask_file):
                        logger.warning(
                            f"Mask file not found for: {osp.basename(image_file)}"
                        )
                        continue

                    output_file = osp.join(output, base_name + xlabel_ext)
                    converter.mask_to_custom(
                        mask_file, output_file, image_file, mapping_table
                    )
                    count += 1

                print(
                    colored(
                        f"✓ Converted {count} mask files to XLABEL format: {output}",
                        "green",
                    )
                )

            elif task == "mot2xlabel":
                if not labels:
                    raise ValueError(
                        "--labels is required (path to MOT annotation file)"
                    )
                if not osp.exists(labels):
                    raise FileNotFoundError(f"MOT file not found: {labels}")

                if not images:
                    raise ValueError(
                        "--images is required (path to video frames directory)"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not classes_file:
                    raise ValueError(
                        "--classes is required for MOT conversion"
                    )

                if not output:
                    output = images
                    logger.warning(
                        f"--output not specified, using image directory: {output}"
                    )
                os.makedirs(output, exist_ok=True)

                print(colored("Converting MOT format to XLABEL...", "cyan"))
                converter.mot_to_custom(labels, output, images)

                output_files = glob(osp.join(output, "*.json"))
                print(
                    colored(
                        f"✓ Converted MOT to {len(output_files)} XLABEL files: {output}",
                        "green",
                    )
                )

            elif task == "odvg2xlabel":
                if not labels:
                    raise ValueError(
                        "--labels is required (path to ODVG JSONL file)"
                    )
                if not osp.exists(labels):
                    raise FileNotFoundError(f"ODVG file not found: {labels}")

                if not output:
                    output = osp.dirname(labels) or "."
                    logger.warning(
                        f"--output not specified, using labels directory: {output}"
                    )
                os.makedirs(output, exist_ok=True)

                print(colored("Converting ODVG format to XLABEL...", "cyan"))
                converter.odvg_to_custom(labels, output)

                output_files = glob(osp.join(output, "*.json"))
                print(
                    colored(
                        f"✓ Converted ODVG to {len(output_files)} XLABEL files: {output}",
                        "green",
                    )
                )

            elif task == "ppocr2xlabel":
                if not labels:
                    raise ValueError(
                        "--labels is required (path to PaddleOCR annotation file)"
                    )
                if not osp.exists(labels):
                    raise FileNotFoundError(
                        f"PaddleOCR file not found: {labels}"
                    )

                if not images:
                    raise ValueError(
                        "--images is required (path to images directory)"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not mode:
                    raise ValueError("--mode is required for PPOCR conversion")
                if mode not in PPOCR_TASK_MODES:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be one of {PPOCR_TASK_MODES}"
                    )

                if not output:
                    output = images
                    logger.warning(
                        f"--output not specified, using image directory: {output}"
                    )
                os.makedirs(output, exist_ok=True)

                print(
                    colored(
                        f"Converting PaddleOCR {mode} format to XLABEL...",
                        "cyan",
                    )
                )
                converter.ppocr_to_custom(labels, output, images, mode)

                output_files = glob(osp.join(output, "*.json"))
                print(
                    colored(
                        f"✓ Converted PaddleOCR to {len(output_files)} XLABEL files: {output}",
                        "green",
                    )
                )

            elif task == "vlmr12xlabel":
                if not images:
                    raise ValueError(
                        "--images is required for vlmr1 conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for vlmr1 conversion"
                    )
                os.makedirs(output, exist_ok=True)

                count = 0
                image_files = get_image_files(images)
                for image_file in tqdm(
                    image_files, desc="Converting VLM-R1 to XLABEL"
                ):
                    base_name = osp.splitext(osp.basename(image_file))[0]
                    label_file = osp.join(labels, base_name + ".txt")
                    if not osp.exists(label_file):
                        logger.warning(
                            f"Label file not found for: {osp.basename(image_file)}"
                        )
                        continue

                    input_data = converter.read_json(label_file)
                    output_file = osp.join(output, base_name + xlabel_ext)
                    converter.vlm_r1_ovd_to_custom(
                        input_data, output_file, image_file
                    )
                    count += 1

                print(
                    colored(
                        f"✓ Converted {count} VLM-R1 files to XLABEL format: {output}",
                        "green",
                    )
                )

        elif "xlabel2" in task:

            if task == "xlabel2yolo":
                if not images:
                    raise ValueError(
                        "--images is required for XLABEL to YOLO conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not mode:
                    raise ValueError("--mode is required for YOLO conversion")
                if mode not in YOLO_TASK_MODES:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be one of {YOLO_TASK_MODES}"
                    )
                if mode in ["detect", "segment", "obb"] and not classes_file:
                    raise ValueError(
                        f"--classes is required for yolo {mode} conversion"
                    )
                elif mode == "pose" and not pose_cfg_file:
                    raise ValueError(
                        "--pose-cfg is required for yolo pose conversion"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not output:
                    output = images
                    logger.warning(
                        f"--output not specified, using image directory: {output}"
                    )
                os.makedirs(output, exist_ok=True)

                mode_map = {
                    "detect": "hbb",
                    "segment": "seg",
                    "obb": "obb",
                    "pose": "pose",
                }
                yolo_mode = mode_map[mode]

                count = 0
                image_files = get_image_files(images)
                for image_file in tqdm(
                    image_files, desc=f"Converting XLABEL to YOLO {mode}"
                ):
                    label_file = find_matching_file(
                        image_file, labels, xlabel_ext
                    )
                    if label_file is None:
                        if skip_empty_files:
                            continue

                        base_name = osp.splitext(osp.basename(image_file))[0]
                        label_file = osp.join(
                            labels, f"{base_name}{xlabel_ext}"
                        )

                    output_file = osp.join(
                        output,
                        osp.splitext(osp.basename(image_file))[0] + yolo_ext,
                    )
                    is_empty = converter.custom_to_yolo(
                        label_file, output_file, yolo_mode, skip_empty_files
                    )
                    if not is_empty or not skip_empty_files:
                        count += 1

                print(
                    colored(
                        f"✓ Converted {count} XLABEL files to YOLO {mode} format: {output}",
                        "green",
                    )
                )

            elif task == "xlabel2voc":
                if not images:
                    raise ValueError(
                        "--images is required for XLABEL to VOC conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not mode:
                    raise ValueError("--mode is required for VOC conversion")
                if mode not in VOC_TASK_MODES:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be one of {VOC_TASK_MODES}"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not output:
                    raise ValueError("--output is required for VOC conversion")
                os.makedirs(output, exist_ok=True)

                voc_mode = "rectangle" if mode == "detect" else "polygon"

                count = 0
                image_files = get_image_files(images)
                for image_file in tqdm(
                    image_files, desc=f"Converting XLABEL to VOC {mode}"
                ):
                    label_file = find_matching_file(
                        image_file, labels, xlabel_ext
                    )
                    if label_file is None:
                        if skip_empty_files:
                            continue

                        base_name = osp.splitext(osp.basename(image_file))[0]
                        label_file = osp.join(
                            labels, f"{base_name}{xlabel_ext}"
                        )

                    output_file = osp.join(
                        output,
                        osp.splitext(osp.basename(image_file))[0] + voc_ext,
                    )
                    is_empty = converter.custom_to_voc(
                        image_file,
                        label_file,
                        output_file,
                        voc_mode,
                        skip_empty_files,
                    )
                    if not is_empty or not skip_empty_files:
                        count += 1

                print(
                    colored(
                        f"✓ Converted {count} XLABEL files to VOC {mode} format: {output}",
                        "green",
                    )
                )

            elif task == "xlabel2coco":
                if not images:
                    raise ValueError(
                        "--images is required for XLABEL to COCO conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not mode:
                    raise ValueError("--mode is required for COCO conversion")
                if mode not in COCO_TASK_MODES:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be one of {COCO_TASK_MODES}"
                    )
                if mode in ["detect", "segment"] and not classes_file:
                    raise ValueError(
                        f"--classes is required for coco {mode} conversion"
                    )
                elif mode == "pose" and not pose_cfg_file:
                    raise ValueError(
                        "--pose-cfg is required for coco pose conversion"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for COCO conversion"
                    )
                os.makedirs(output, exist_ok=True)

                coco_mode_map = {
                    "detect": "rectangle",
                    "segment": "polygon",
                    "pose": "pose",
                }
                coco_mode = coco_mode_map[mode]

                print(
                    colored(
                        f"Converting XLABEL to COCO {mode} format...", "cyan"
                    )
                )
                image_files = get_image_files(images)
                converter.custom_to_coco(
                    image_files, labels, output, coco_mode
                )

                output_files = glob(osp.join(output, "*.json"))
                print(
                    colored(
                        f"✓ Converted XLABEL to COCO {mode} format: {output}",
                        "green",
                    )
                )

            elif task == "xlabel2dota":
                if not images:
                    raise ValueError(
                        "--images is required for XLABEL to DOTA conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for DOTA conversion"
                    )
                os.makedirs(output, exist_ok=True)

                count = 0
                image_files = get_image_files(images)
                for image_file in tqdm(
                    image_files, desc="Converting XLABEL to DOTA"
                ):
                    label_file = find_matching_file(
                        image_file, labels, xlabel_ext
                    )
                    if not label_file:
                        logger.warning(
                            f"Label file not found for: {osp.basename(image_file)}"
                        )
                        continue

                    output_file = osp.join(
                        output,
                        osp.splitext(osp.basename(image_file))[0] + dota_ext,
                    )
                    converter.custom_to_dota(label_file, output_file)
                    count += 1

                print(
                    colored(
                        f"✓ Converted {count} XLABEL files to DOTA format: {output}",
                        "green",
                    )
                )

            elif task == "xlabel2mask":
                if not images:
                    raise ValueError(
                        "--images is required for XLABEL to mask conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not mapping_file:
                    raise ValueError(
                        "--mapping is required (path to mapping table file)"
                    )
                if not osp.exists(mapping_file):
                    raise FileNotFoundError(
                        f"Mapping file not found: {mapping_file}"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for mask conversion"
                    )
                os.makedirs(output, exist_ok=True)

                count = 0
                mapping_table = converter.read_json(mapping_file)
                image_files = get_image_files(images)
                for image_file in tqdm(
                    image_files, desc="Converting XLABEL to mask"
                ):
                    base_name = osp.splitext(osp.basename(image_file))[0]
                    label_file = osp.join(labels, base_name + xlabel_ext)
                    output_file = osp.join(output, base_name + mask_ext)

                    converter.custom_to_mask(
                        label_file, output_file, mapping_table
                    )
                    count += 1

                print(
                    colored(
                        f"✓ Converted {count} XLABEL files to mask format: {output}",
                        "green",
                    )
                )

            elif task == "xlabel2mot":
                if not labels:
                    raise ValueError(
                        "--labels is required (path to XLABEL directory)"
                    )
                if not osp.exists(labels):
                    raise FileNotFoundError(
                        f"Labels directory not found: {labels}"
                    )

                if not classes_file:
                    raise ValueError(
                        "--classes is required for MOT conversion"
                    )

                if not output:
                    raise ValueError("--output is required for MOT conversion")
                os.makedirs(output, exist_ok=True)

                print(colored("Converting XLABEL to MOT format...", "cyan"))
                converter.custom_to_mot(labels, output)
                print(
                    colored(
                        f"✓ Converted XLABEL to MOT format: {output}", "green"
                    )
                )

            elif task == "xlabel2mots":
                if not labels:
                    raise ValueError(
                        "--labels is required (path to XLABEL directory)"
                    )
                if not osp.exists(labels):
                    raise FileNotFoundError(
                        f"Labels directory not found: {labels}"
                    )

                if not classes_file:
                    raise ValueError(
                        "--classes is required for MOTS conversion"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for MOTS conversion"
                    )
                os.makedirs(output, exist_ok=True)

                print(colored("Converting XLABEL to MOTS format...", "cyan"))
                converter.custom_to_mots(labels, output)
                print(
                    colored(
                        f"✓ Converted XLABEL to MOTS format: {output}", "green"
                    )
                )

            elif task == "xlabel2odvg":
                if not images:
                    raise ValueError(
                        "--images is required for XLABEL to ODVG conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not classes_file:
                    raise ValueError(
                        "--classes is required for ODVG conversion"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for ODVG conversion"
                    )
                os.makedirs(output, exist_ok=True)

                print(colored("Converting XLABEL to ODVG format...", "cyan"))
                image_files = get_image_files(images)
                converter.custom_to_odvg(image_files, labels, output)
                print(
                    colored(
                        f"✓ Converted XLABEL to ODVG format: {output}", "green"
                    )
                )

            elif task == "xlabel2vlmr1":
                if not images:
                    raise ValueError(
                        "--images is required for XLABEL to VLM-R1 conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not output:
                    raise ValueError("--output is required (JSONL file path)")

                output_dir = osp.dirname(output)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

                print(colored("Converting XLABEL to VLM-R1 format...", "cyan"))
                image_files = get_image_files(images)
                converter.custom_to_vlm_r1_ovd(image_files, labels, output)
                print(
                    colored(
                        f"✓ Converted XLABEL to VLM-R1 format: {output}",
                        "green",
                    )
                )

            elif task == "xlabel2ppocr":
                if not images:
                    raise ValueError(
                        "--images is required for XLABEL to PaddleOCR conversion"
                    )
                if not osp.exists(images):
                    raise FileNotFoundError(
                        f"Image directory not found: {images}"
                    )

                if not labels:
                    labels = images
                    logger.warning(
                        f"--labels not specified, using image directory: {labels}"
                    )

                if not mode:
                    raise ValueError("--mode is required for PPOCR conversion")
                if mode not in PPOCR_TASK_MODES:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be one of {PPOCR_TASK_MODES}"
                    )

                if not output:
                    raise ValueError(
                        "--output is required for PaddleOCR conversion"
                    )
                os.makedirs(output, exist_ok=True)

                if mode == "rec":
                    crop_img_path = osp.join(output, "crop_img")
                    os.makedirs(crop_img_path, exist_ok=True)

                print(
                    colored(
                        f"Converting XLABEL to PaddleOCR {mode} format...",
                        "cyan",
                    )
                )
                image_files = get_image_files(images)
                class_set = set()
                for image_file in tqdm(
                    image_files, desc=f"Converting XLABEL to PaddleOCR {mode}"
                ):
                    base_name = osp.splitext(osp.basename(image_file))[0]
                    label_file = osp.join(labels, base_name + xlabel_ext)
                    result = converter.custom_to_ppocr(
                        image_file, label_file, output, mode
                    )
                    if mode == "kie" and result:
                        class_set.update(result)

                if mode == "kie" and class_set:
                    class_list_file = osp.join(output, "class_list.txt")
                    with open(class_list_file, "w", encoding="utf-8") as f:
                        for cls in sorted(class_set):
                            f.write(f"{cls}\n")
                    print(
                        colored(
                            f"✓ Saved class list to: {class_list_file}",
                            "green",
                        )
                    )

                print(
                    colored(
                        f"✓ Converted XLABEL to PaddleOCR {mode} format: {output}",
                        "green",
                    )
                )

    except Exception as e:
        print(colored(f"✗ Error: {e}", "red"), file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_conversion()
