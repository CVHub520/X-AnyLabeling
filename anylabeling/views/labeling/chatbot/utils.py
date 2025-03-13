from openai import OpenAI


def get_models_list(base_url: str, api_key: str) -> list:
    """Get models list from the API"""
    client = OpenAI(base_url=base_url, api_key=api_key)
    return [model.id for model in client.models.list()]


def set_icon_path(icon_name: str) -> str:
    """Set the path to the icon"""
    return f"anylabeling/resources/icons/{icon_name}.svg"
