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

MAIN_TASK_LIST = all_task_names = [name for task_info in TASK_REGISTRY.values() for name in task_info.keys()]
COMPLETE_TASK_LIST = [name for task_info in TASK_REGISTRY.values() for name in task_info.keys()]


def get_task_info(task_name: str) -> (str, Dict):
    task_registry = yaml.safe_load(open(task_registry_path))
    for group, tasks in task_registry.items():
        for name, info in tasks.items():
            if name == task_name:
                return group, info
    raise ValueError(f"No config for the task {task_name} found!")

def get_task_name_from_alias(task_alias:str) -> str:
    for task_group, task in TASK_REGISTRY.items():
        for task_name, task_info in task.items():
            if task_info['alias'] == task_alias:
                return task_name

def get_alias(task_name:str) -> str:
    _, info = get_task_info(task_name)
    return info["alias"]

def get_baseline(task_name:str) -> str:
    _, info = get_task_info(task_name)
    return info["random_baseline"]

def get_capability_group_from_task_name(task_name:str) -> str:
    _, task_info = get_task_info(task_name)
    if task_info["category"] in ["interactive", "massive", "extra"]:
        return task_info["category"]
    else:
        for capability_group, capability in CAPABILITY_REGISTRY.items():
            for capability_name, capability_info in capability.items():
                if capability_name == task_info["category"]:
                    return capability_group

def get_capability_group_from_alias(task_alias:str) -> str:
    task_name = get_task_name_from_alias(task_alias)
    return get_capability_group_from_task_name(task_name)

def get_capability_alias(capability_name):
    for capability_group, capability_dict in CAPABILITY_REGISTRY.items():
        for capability, capability_info in capability_dict.items():
            if capability == capability_name:
                return capability_info['alias']


def get_task_backend(task_name: str) -> str:
    for group, tasks in TASK_REGISTRY.items():
        for name, info in tasks.items():
            if name == task_name:
                return info["backend"]
    return None
