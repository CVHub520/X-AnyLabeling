import os
import yaml
import requests


def url_exists(url, timeout=5):
    try:
        response = requests.head(url, timeout=timeout)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def main(models_yaml_path):
    # Load the models.yaml file
    with open(models_yaml_path, 'r') as file:
        models = yaml.safe_load(file)

    # Get the directory containing the YAML file
    yaml_dir = os.path.dirname(models_yaml_path)

    for model in models:
        config_file = model.get('config_file')
        if config_file:
            # Construct the full path to the YAML file
            yaml_file_path = os.path.join(
                yaml_dir, config_file.lstrip(':/')
            )

            # Load the YAML configuration
            with open(yaml_file_path, 'r') as file:
                yaml_config = yaml.safe_load(file)

            # Extract and test URL links
            for key, value in yaml_config.items():
                if isinstance(value, str) and 'path' in key:
                    if not url_exists(value):
                        print(f"⚠️ URL {key}: {value} is not "
                              f"accessible for model {model['model_name']}")
                    else:
                        print(f"✅ URL {key}: {value}")

if __name__ == '__main__':
    models_yaml_path = "anylabeling/configs/auto_labeling/models.yaml"
    main(models_yaml_path)
