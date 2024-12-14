import yaml
import inspect
import importlib.util
from frameworks.playeval_framework import framework_root, playeval_logger
from frameworks.playeval_framework.tasks.task import Task
from typing import Dict, Any

# Function to dynamically import a module
def import_module_from_path(file_path):
    module_name = file_path.stem  # Get the module name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_task(task_name:str):
    task_root = framework_root / "tasks"
    # Iterate over all files ending with "_task.py"
    for file_path in task_root.rglob("*_task.py"):
        try:
            # Dynamically import the Python file as a module
            module = import_module_from_path(file_path)

            # Iterate over all classes defined in the module
            for name, cls in inspect.getmembers(module, inspect.isclass):
                # Check if the class inherits from Task
                if issubclass(cls, Task) and cls is not Task:
                    # Create an instance of the class to check `self.task_name`
                    try:
                        instance = cls()  # Ensure the class has a no-argument constructor
                        if hasattr(instance, "task_name") and instance.task_name == task_name:
                            return instance
                            playeval_logger.info(f"Match found: {cls.__name__} in {file_path}")
                    except TypeError as e:
                        playeval_logger.error(f"Could not instantiate {cls.__name__}: {e}")
        except Exception as e:
            playeval_logger.error(f"Error processing {file_path}: {e}")

def get_task_config(task:str) -> Dict[str, Any]:
    data = {}
    try:
        task_config_root = framework_root / "tasks" / task
        for yaml_file in task_config_root.rglob("*.yaml"):
            if yaml_file.is_file():
                try:
                    with yaml_file.open("r") as file:
                        data = yaml.safe_load(file)  # Load YAML content

                    # Check if 'task_name' exists and matches the desired value
                    if isinstance(data, dict) and data.get("task_name") == task:
                        return data
                except yaml.YAMLError as e:
                    print(f"Error parsing {yaml_file}: {e}")
    except ValueError:
        playeval_logger.error(f"There was an error trying to retrieve the config file for task {task}.")
    if not data:
        raise Exception(f"No config file for the task '{task}' has been found, or it's empty.")