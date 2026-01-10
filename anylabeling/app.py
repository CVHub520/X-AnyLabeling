import os

# Temporary fix for: bus error
# Source: https://stackoverflow.com/questions/73072612/
# why-does-np-linalg-solve-raise-bus-error-when-running-on-its-own-thread-mac-m1
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress ICC profile warnings
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.gui.icc=false"

import argparse
import codecs
import logging

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

if any(
    arg.startswith(("parent_pid=", "pipe_handle=", "--multiprocessing-"))
    for arg in sys.argv[1:]
):
    sys.exit(0)

import yaml
from PyQt5 import QtCore, QtWidgets

from anylabeling.app_info import (
    __appname__,
    __version__,
    __url__,
    CLI_HELP_MSG,
)
from anylabeling.config import get_config
from anylabeling import config as anylabeling_config
from anylabeling.views.mainwindow import MainWindow
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils import new_icon, gradient_text
from anylabeling.views.labeling.utils.update_checker import (
    check_for_updates_async,
)

# NOTE: Do not remove this import, it is required for loading translations
from anylabeling.resources import resources


def main():
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    filtered_argv = []
    for arg in sys.argv[1:]:
        if not (
            arg.startswith("parent_pid=")
            or arg.startswith("pipe_handle=")
            or arg.startswith("--multiprocessing-")
        ):
            filtered_argv.append(arg)
    sys.argv = [sys.argv[0]] + filtered_argv

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command", help="available commands"
    )
    subparsers.add_parser("help", help="show help message")
    subparsers.add_parser(
        "checks", help="display system and package information"
    )
    subparsers.add_parser("version", help="show version information")
    subparsers.add_parser("config", help="show config file path")

    convert_parser = subparsers.add_parser(
        "convert", help="run conversion tasks"
    )
    convert_parser.add_argument(
        "--task",
        type=str,
        help="conversion task name (e.g., yolo2xlabel, xlabel2yolo)",
    )
    convert_parser.add_argument(
        "--images", type=str, help="image directory path"
    )
    convert_parser.add_argument(
        "--labels", type=str, help="label directory path"
    )
    convert_parser.add_argument(
        "--output", type=str, help="output directory path"
    )
    convert_parser.add_argument(
        "--classes", type=str, help="classes file path"
    )
    convert_parser.add_argument(
        "--pose-cfg", type=str, help="pose configuration file path"
    )
    convert_parser.add_argument("--mode", type=str, help="conversion mode")
    convert_parser.add_argument(
        "--mapping", type=str, help="mapping table file path"
    )
    convert_parser.add_argument(
        "--skip-empty-files",
        action="store_true",
        help="skip creating empty output files, only support `xlabel2yolo` and `xlabel2voc` tasks",
    )

    parser.add_argument(
        "--reset-config", action="store_true", help="reset qt config"
    )
    parser.add_argument(
        "--logger-level",
        default="info",
        choices=["debug", "info", "warning", "fatal", "error"],
        help="logger level",
    )
    parser.add_argument(
        "--no-auto-update-check",
        action="store_true",
        help="disable automatic update check on startup",
    )
    parser.add_argument(
        "--qt-platform",
        help=(
            "Force Qt platform plugin (e.g., 'xcb', 'wayland'). "
            "If not specified, Qt will auto-detect the platform."
        ),
        default=None,
    )
    parser.add_argument(
        "--filename",
        nargs="?",
        help=(
            "image or label filename; "
            "If a directory path is passed in, the folder will be loaded automatically"
        ),
    )
    parser.add_argument(
        "--output",
        "-O",
        "-o",
        help=(
            "output file or directory (if it ends with .json it is "
            "recognized as file, else as directory)"
        ),
    )
    default_config_file = os.path.join(
        os.path.expanduser("~"), ".xanylabelingrc"
    )
    parser.add_argument(
        "--config",
        dest="config",
        help=(
            "config file or yaml-format string (default:"
            f" {default_config_file})"
        ),
        default=default_config_file,
    )
    # config for the gui
    parser.add_argument(
        "--nodata",
        dest="store_data",
        action="store_false",
        help="stop storing image data to JSON file",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--autosave",
        dest="auto_save",
        action="store_true",
        help="auto save",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--nosortlabels",
        dest="sort_labels",
        action="store_false",
        help="stop sorting labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--flags",
        help="comma separated list of flags OR file containing flags",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labelflags",
        dest="label_flags",
        help=r"yaml string of label specific flags OR file containing json "
        r"string of label specific flags (ex. {person-\d+: [male, tall], "
        r"dog-\d+: [black, brown, white], .*: [occluded]})",  # NOQA
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labels",
        help="comma separated list of labels OR file containing labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validatelabel",
        dest="validate_label",
        choices=["exact"],
        help="label validation types",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-prev",
        action="store_true",
        help="keep annotation of previous frame",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="epsilon to find nearest vertex on canvas",
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    special = {
        "help": lambda args: print(CLI_HELP_MSG),
        "checks": lambda args: __import__(
            "anylabeling.views.common.checks", fromlist=["run_checks"]
        ).run_checks(),
        "version": lambda args: print(__version__),
        "config": lambda args: print(
            os.path.join(os.path.expanduser("~"), ".xanylabelingrc")
        ),
        "convert": lambda args: __import__(
            "anylabeling.views.common.converter",
            fromlist=["handle_convert_command"],
        ).handle_convert_command(args),
    }

    if args.command and args.command in special:
        special[args.command](args)
        return

    if hasattr(args, "flags"):
        if os.path.isfile(args.flags):
            with codecs.open(args.flags, "r", encoding="utf-8") as f:
                args.flags = [line.strip() for line in f if line.strip()]
        else:
            args.flags = [line for line in args.flags.split(",") if line]

    if hasattr(args, "labels"):
        if os.path.isfile(args.labels):
            with codecs.open(args.labels, "r", encoding="utf-8") as f:
                args.labels = [line.strip() for line in f if line.strip()]
        else:
            args.labels = [line for line in args.labels.split(",") if line]

    if hasattr(args, "label_flags"):
        if os.path.isfile(args.label_flags):
            with codecs.open(args.label_flags, "r", encoding="utf-8") as f:
                args.label_flags = yaml.safe_load(f)
        else:
            args.label_flags = yaml.safe_load(args.label_flags)

    config_from_args = args.__dict__
    config_from_args.pop("command", None)
    reset_config = config_from_args.pop("reset_config")
    filename = config_from_args.pop("filename")
    output = config_from_args.pop("output")
    config_file_or_yaml = config_from_args.pop("config")
    logger_level = config_from_args.pop("logger_level")
    no_auto_update_check = config_from_args.pop("no_auto_update_check", False)
    qt_platform = config_from_args.pop("qt_platform", None)

    logger.setLevel(getattr(logging, logger_level.upper()))
    logger.info(
        f"🚀 {gradient_text(f'X-AnyLabeling v{__version__} launched!')}"
    )
    logger.info(f"⭐ If you like it, give us a star: {__url__}")
    if qt_platform:
        os.environ["QT_QPA_PLATFORM"] = qt_platform
        logger.info(f"🖥️ Using Qt platform: {qt_platform}")

    anylabeling_config.current_config_file = config_file_or_yaml
    config = get_config(config_file_or_yaml, config_from_args, show_msg=True)

    if not config["labels"] and config["validate_label"]:
        logger.error(
            "--labels must be specified with --validatelabel or "
            "validate_label: exact in the config file "
            "(ex. ~/.xanylabelingrc)."
        )
        sys.exit(1)

    output_file = None
    output_dir = None
    if output is not None:
        if output.endswith(".json"):
            output_file = output
        else:
            output_dir = output

    language = config.get("language", QtCore.QLocale.system().name())
    translator = QtCore.QTranslator()
    loaded_language = translator.load(
        ":/languages/translations/" + language + ".qm"
    )
    # Enable scaling for high dpi screens
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_EnableHighDpiScaling, True
    )  # enable highdpi scaling
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_UseHighDpiPixmaps, True
    )  # use highdpi icons
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

    app = QtWidgets.QApplication(sys.argv)
    app.processEvents()

    app.setApplicationName(__appname__)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(new_icon("icon"))
    if loaded_language:
        app.installTranslator(translator)
    else:
        logger.warning(
            f"Failed to load translation for {language}. "
            "Using default language.",
        )
    win = MainWindow(
        app,
        config=config,
        filename=filename,
        output_file=output_file,
        output_dir=output_dir,
    )

    if reset_config:
        logger.info(f"Resetting Qt config: {win.settings.fileName()}")
        win.settings.clear()
        sys.exit(0)

    if not no_auto_update_check:

        def delayed_update_check():
            check_for_updates_async(timeout=5)

        QtCore.QTimer.singleShot(2000, delayed_update_check)

    win.showMaximized()
    win.raise_()
    sys.exit(app.exec())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    main()
