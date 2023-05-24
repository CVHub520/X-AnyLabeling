import os

supported_languages = ["en_US", "zh_CN"]

for language in supported_languages:
    # Compile the .ts file into a .qm file
    command = f"lrelease anylabeling/resources/translations/{language}.ts"
    os.system(command)

# Generate resources
command = "pyrcc5 -o anylabeling/resources/resources.py \
    anylabeling/resources/resources.qrc"
os.system(command)
