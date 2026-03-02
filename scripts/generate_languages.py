import os
import glob
import sys
import shutil
import subprocess
from PyQt6 import QtCore


def compile_resources(output: str, qrc: str) -> None:
    """Compile a .qrc file to a PyQt6-compatible resources.py."""

    def normalize_imports(path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("from PySide6", "from PyQt6")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    commands = [
        ([sys.executable, "-m", "PyQt6.pyrcc_main", "-o", output, qrc], False),
        (["pyrcc6", "-o", output, qrc], False),
        (["pyside6-rcc", "-o", output, qrc], True),
        (["rcc", "-g", "python", "-o", output, qrc], True),
    ]
    lrelease = shutil.which("lrelease")
    if lrelease:
        sibling_rcc = os.path.join(os.path.dirname(lrelease), "rcc")
        commands.append(
            ([sibling_rcc, "-g", "python", "-o", output, qrc], True)
        )
    for command, needs_rewrite in commands:
        executable = command[0]
        if executable != sys.executable and not shutil.which(executable):
            continue
        result = subprocess.run(command, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            continue
        if needs_rewrite:
            normalize_imports(output)
        return
    print(
        "Error: no Qt resource compiler found. Tried python -m PyQt6.pyrcc_main, pyrcc6, pyside6-rcc, rcc -g python, and lrelease-sibling rcc."
    )


supported_languages = ["en_US", "zh_CN"]
translations_path = "anylabeling/resources/translations"

for language in supported_languages:
    # Scan all .py files in the project directory and its subdirectories
    py_files = glob.glob(os.path.join("**", "*.py"), recursive=True)

    # Create a QTranslator object to generate the .ts file
    translator = QtCore.QTranslator()

    # Translate all .ui files into .py files
    ui_files = glob.glob(os.path.join("**", "*.ui"), recursive=True)
    for ui_file in ui_files:
        py_file = os.path.splitext(ui_file)[0] + "_ui.py"
        command = f"pyuic6 -x {ui_file} -o {py_file}"
        os.system(command)

    # Extract translations from the .py file
    command = f"pylupdate6 --no-obsolete {' '.join(py_files)} -ts {translations_path}/{language}.ts"
    os.system(command)

    # Compile the .ts file into a .qm file
    command = f"lrelease {translations_path}/{language}.ts"
    os.system(command)

compile_resources(
    output="anylabeling/resources/resources.py",
    qrc="anylabeling/resources/resources.qrc",
)
