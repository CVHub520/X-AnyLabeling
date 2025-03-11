from openai import OpenAI


def get_models_list(base_url: str, api_key: str) -> list:
    """Get models list from the API"""
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Get models list
    models_list = client.models.list()

    # Process model data
    models_data = [model.id for model in models_list.data]

    return models_data


def set_icon_path(icon_name: str) -> str:
    """Set the path to the icon"""
    return f"anylabeling/resources/icons/{icon_name}.svg"
