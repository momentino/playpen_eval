import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any


#playpen_eval_logger = logging.getLogger("playpen_eval_logger")

project_root = Path(os.path.abspath(__file__)).parent.parent
config_path = project_root / "config"
task_registry_path = config_path / "task_registry.yaml"
model_registry_path = config_path / "model_registry.yaml"

def get_model_registry():
    model_registry = yaml.safe_load(open(model_registry_path))
    return model_registry

def get_task_registry():
    task_registry = yaml.safe_load(open(task_registry_path))
    return task_registry

def get_task_info(task_name: str) -> (str, Dict):
    task_registry = yaml.safe_load(open(task_registry_path))
    for group, tasks in task_registry.items():
        for name, info in tasks.items():
            if name == task_name:
                return group, info
    raise ValueError(f"No config for the task {task_name} found!")

def get_alias(task_name:str) -> str:
    _, info = get_task_info(task_name)
    return info["alias"]

def get_baseline(task_name:str) -> str:
    _, info = get_task_info(task_name)
    return info["random_baseline"]

def get_functional_group_from_alias(task_name:str, tasks_info: Dict) -> str:
    for group, tasks in tasks_info.items():
        for name, info in tasks.items():
            if info["alias"] == task_name:
                return info["functional_groups"]

def get_task_backend(task_name: str, task_registry: Dict) -> str:
    for group, tasks in task_registry.items():
        for name, info in tasks.items():
            if name == task_name:
                return info["backend"]
    return None
