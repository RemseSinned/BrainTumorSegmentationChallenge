import yaml
from pathlib import Path


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
