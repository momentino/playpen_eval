import os
import yaml
from pathlib import Path
from typing import Dict
project_root = Path(os.path.abspath(__file__)).parent.parent
config_path = project_root / "config"
task_registry_path = config_path / "task_registry.yaml"
model_registry_path = config_path / "model_registry.yaml"
capability_registry_path = config_path / "capability_registry.yaml"

MODEL_REGISTRY = yaml.safe_load(open(model_registry_path))
TASK_REGISTRY = yaml.safe_load(open(task_registry_path))
CAPABILITY_REGISTRY = yaml.safe_load(open(capability_registry_path))


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

def get_capability_group_from_alias(task_name:str) -> str:
    _, task_info = get_task_info(task_name)
    for group, tasks in tasks_info.items():
        for name, info in tasks.items():
            if info["alias"] == task_name:
                if info["category"] in ["interactive","massive"]:
                    return info["category"]
                else:
                    for capability_group, capability in CAPABILITY_REGISTRY.items():
                        if capability == info["category"]:
                            return capability_group

def get_task_backend(task_name: str) -> str:
    for group, tasks in TASK_REGISTRY.items():
        for name, info in tasks.items():
            if name == task_name:
                return info["backend"]
    return None
