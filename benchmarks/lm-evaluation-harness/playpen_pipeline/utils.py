import yaml
import os
import json
from pathlib import Path



def load_yaml_config(file_path: Path) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)