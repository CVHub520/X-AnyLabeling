import os
import glob
from PyQt5 import QtCore

supported_languages = ["en_US", "zh_CN"]

for language in supported_languages:
    # Scan all .py files in the project directory and its subdirectories
    py_files = glob.glob(os.path.join("**", "*.py"), recursive=True)

    # Create a QTranslator object to generate the .ts file
    translator = QtCore.QTranslator()

    # Translate all .ui files into .py files
    ui_files = glob.glob(os.path.join("**", "*.ui"), recursive=True)
    for ui_file in ui_files:
        py_file = os.path.splitext(ui_file)[0] + "_ui.py"
        command = f"pyuic5 -x {ui_file} -o {py_file}"
        os.system(command)

    # Extract translations from the .py file
    translations_path = "anylabeling/resources/translations"
    command = f"pylupdate5 {' '.join(py_files)} -ts {translations_path}/{language}.ts"
    os.system(command)

    # Compile the .ts file into a .qm file
    command = f"lrelease {translations_path}/{language}.ts"
    os.system(command)

# Generate resources
command = "pyrcc5 -o anylabeling/resources/resources.py \
    anylabeling/resources/resources.qrc"
os.system(command)
