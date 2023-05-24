import yaml
import os
import pathlib
import zipfile
from urllib.parse import urlparse


output_path = "zipped_models/"
model_config_path = "anylabeling/configs/auto_labeling/"
model_list_path = "anylabeling/configs/auto_labeling/models.yaml"
model_list = yaml.load(open(model_list_path, "r"), Loader=yaml.FullLoader)

# Create output path
pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)


def get_filename_from_url(url):
    a = urlparse(url)
    return os.path.basename(a.path)


for model in model_list:
    model_name = model["model_name"]
    config_file = model["config_file"]
    print(f"Zipping {model_name}...")

    # Get download links
    download_links = []
    model_config = yaml.load(
        open(model_config_path + config_file, "r"), Loader=yaml.FullLoader
    )
    if model_config["type"] == "segment_anything":
        download_links.append(model_config["encoder_model_path"])
        download_links.append(model_config["decoder_model_path"])
    else:
        download_links.append(model_config["model_path"])

    model_output_path = os.path.join(output_path, model_name)
    pathlib.Path(model_output_path).mkdir(parents=True, exist_ok=True)

    # Save model config
    # Rewrite model's urls
    if model_config["type"] == "segment_anything":
        model_config["encoder_model_path"] = get_filename_from_url(
            model_config["encoder_model_path"]
        )
        model_config["decoder_model_path"] = get_filename_from_url(
            model_config["decoder_model_path"]
        )
    else:
        model_config["model_path"] = get_filename_from_url(
            model_config["model_path"]
        )
    with open(os.path.join(model_output_path, "config.yaml"), "w") as f:
        yaml.dump(model_config, f)

    # Download models
    for link in download_links:
        os.system(f"wget -P {model_output_path} {link}")

    # Zip model
    with zipfile.ZipFile(
        os.path.join(output_path, f"{model_name}.zip"), "w"
    ) as zip:
        for root, _, files in os.walk(model_output_path):
            for file in files:
                zip.write(
                    os.path.join(root, file),
                    os.path.relpath(
                        os.path.join(root, file),
                        os.path.join(model_output_path, ".."),
                    ),
                )
    os.system(f"rm -rf {model_output_path}")

print("Done!")
