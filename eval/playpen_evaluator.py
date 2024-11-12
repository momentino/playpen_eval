import os
import json
import sys
from pathlib import Path
from typing import List
from datetime import datetime
from eval import playpen_eval_logger, get_executed_tasks

project_folder = Path(os.path.abspath(__file__)).parent.parent

# Add the path of the submodule folder to sys.path, while renaming the directory for import purposes
submodule_path = os.path.join(project_folder, "benchmarks/static/lm-evaluation-harness")
sys.path.insert(0, submodule_path)

import importlib

# Dynamically import the module with the hyphen in its name
lm_eval = importlib.import_module('benchmarks.static.lm-evaluation-harness.lm_eval')
from lm_eval.tasks import TaskManager



class PlaypenEvaluator:
    def __init__(self):
        pass

    def _save_reports(self) -> None:
        pass

    @staticmethod
    def list_tasks() -> None:
        pass

    @staticmethod
    def run(model_backend: str, model_args: str, tasks: List, device: str, log_samples: bool, trust_remote_code:bool, results_path: Path) -> None:

        model_name_parts = model_args.split(",")
        # Look for the part that starts with "pretrained="
        model_name = next(
            (part.replace("pretrained=", "").replace("/", "__") for part in model_name_parts if "pretrained=" in part),
            None  # Default value if "pretrained=" is not found
        )
        model_results_path = Path(os.path.join(project_folder, results_path)) / model_name
        model_results_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        if trust_remote_code:
            import datasets

            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

            model_args = model_args + ",trust_remote_code=True"
        else:
            model_args = model_args

        task_manager = TaskManager()
        if len(tasks) == 0:
            if "all" in tasks[0]:
                tasks = task_manager.all_tasks
            elif "remaning" in tasks[0]:
                # Check for already executed tasks
                executed_tasks, other_tasks = get_executed_tasks(model_results_path, tasks)
                task_manager = TaskManager()
                all_tasks = task_manager.all_tasks
                tasks = [t for t in other_tasks if t in all_tasks]
                unk_tasks = [t for t in other_tasks if t not in all_tasks]
                playpen_eval_logger.info(f"The current model has been already evaluated on the tasks: {executed_tasks}")
                playpen_eval_logger.info(f"Unknown/Not yet implemented tasks: {unk_tasks}")
        playpen_eval_logger.info(f"Now evaluating on {tasks}")

        # Run evaluation for each task
        for task in tasks:
            results = lm_eval.simple_evaluate(
                model=model_backend,
                model_args=model_args,
                tasks=tasks,
                device=device,
                log_samples=log_samples,
            )
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            output_file_path = Path(os.path.join(model_results_path, f"{task}_results{timestamp}.json"))
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w") as file:
                results = json.dumps(str(results))
                json.dump(results, file)

    @staticmethod
    def score() -> None:
        pass