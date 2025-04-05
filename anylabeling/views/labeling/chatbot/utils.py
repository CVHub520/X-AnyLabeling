import json
import os
import time


class EventTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventTracker, cls).__new__(cls)
            cls._instance.counters = {}
            cls._instance.timer = {}
        return cls._instance

    def increment(self, counter_name):
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        self.counters[counter_name] += 1
        if counter_name not in self.timer:
            self.timer[counter_name] = time.time()
        return self.counters[counter_name]

    def get_count(self, counter_name):
        return self.counters.get(counter_name, 0)

    def get_all_counts(self):
        return self.counters.copy()

    def reset(self, counter_name=None):
        if counter_name is None:
            self.counters = {}
            self.timer = {}
        elif counter_name in self.counters:
            self.counters[counter_name] = 0
            self.timer[counter_name] = 0


def load_json(file_path: str) -> dict:
    """Load the json file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, file_path: str):
    """Save the json file"""
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def set_icon_path(icon_name: str, format: str = "svg") -> str:
    """Set the path to the icon

    Args:
        icon_name: Name of the icon file without extension
        format: File format extension (default: 'svg')
    """
    return f"anylabeling/resources/icons/{icon_name}.{format}"
