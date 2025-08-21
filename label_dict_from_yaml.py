import yaml

from typing import Dict

def label_dict_from_yaml(yaml_path: str = 'config.yaml') -> Dict[str, int]:
    """Returns dict {id: gesture_name}"""
    try:
        with open('yaml_path', 'r') as file:
            return yaml.safe_load(file).get("gestures", {})
    except FileNotFoundError:
        print(f"[WARN] File {yaml_path} not found")
    except Exception as e:
        print(f"[ERROR] {e}")
        return {}