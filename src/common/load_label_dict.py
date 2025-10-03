import yaml
from typing import Dict

def load_label_dict(path: str = "config.yaml") -> Dict[int, str]:
    """Load {id: gesture_name} from YAML."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("gestures", {})
    except FileNotFoundError:
        print(f"[WARN] File {path} not found")
        return {}
    except Exception as e:
        print(f"[ERROR] {e}")
        return {}