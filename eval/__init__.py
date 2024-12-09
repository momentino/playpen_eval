import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any

playpen_eval_logger = logging.getLogger("playpen_eval_logger")

def get_playpen_tasks() -> Dict[str, Dict[str, Any]]:
    tasks_file = Path(__file__).parent.parent / "config" / "task_registry.yaml"
    with open(tasks_file, 'r') as file:
        data = yaml.safe_load(file)
    return data['tasks']


def get_executed_tasks(model_results_path: Path, tasks: List[str]) -> (List[str], List[str]):
    executed_tasks = set()
    for json_file in model_results_path.glob("*.json"):
        with open(json_file, "r") as file:
            try:
                data = json.load(file)
                if "results" in data:
                    executed_tasks.update(data["results"].keys())
            except json.JSONDecodeError:
                playpen_eval_logger.warning(f"Warning: {json_file} could not be decoded as JSON.")

    executed_tasks = list(executed_tasks)
    pending_tasks = [task for task in tasks if task not in executed_tasks]

    return executed_tasks, pending_tasks