import os
import json
import sys
import importlib
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime
from eval import playpen_eval_logger, get_executed_tasks, get_playpen_tasks
from utils.utils import custom_json_serializer, prepare_playpen_results




def print_value_types(data):
    # Iterate through each key-value pair in the dictionary
    for key, value in data.items():
        # Print the type of the value
        print(f"Key: {key}, Value Type: {type(value)}")

        # If the value is a dictionary, recurse into it
        if isinstance(value, dict):
            print(f"Recursing into dictionary at key: {key}")
            print_value_types(value)


class PlaypenEvaluator:

    @staticmethod
    def list_tasks() -> None:
        pass

    @staticmethod
    def run(model_backend: str, model_args: str, tasks: List, device: str, trust_remote_code:bool, results_path: Path = "results") -> None:

        model_name_parts = model_args.split(",")
        # Look for the part that starts with "pretrained="
        model_name = next(
            (part.replace("pretrained=", "").replace("/", "__") for part in model_name_parts if "pretrained=" in part),
            None  # Default value if "pretrained=" is not found
        )
        model_harness_results_path = Path(os.path.join(project_folder, results_path)) / "harness" / model_name
        model_harness_results_path.mkdir(parents=True, exist_ok=True)

        model_playpen_results_path = Path(os.path.join(project_folder, results_path)) / "playpen" / model_name
        model_playpen_results_path.mkdir(parents=True, exist_ok=True)

        if trust_remote_code:
            import datasets

            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

            model_args = model_args + ",trust_remote_code=True"
        else:
            model_args = model_args

        playpen_tasks = get_playpen_tasks()
        playpen_task_names = [n for n in playpen_tasks.keys() if playpen_tasks[n]['main_task'] == True]
        if len(tasks) == 1:
            if "all" in tasks[0]:
                tasks = playpen_task_names
            elif "remaining" in tasks[0]:
                # Check for already executed tasks
                executed_tasks, other_tasks = get_executed_tasks(Path(model_harness_results_path), playpen_task_names)
                tasks = other_tasks
                playpen_eval_logger.info(f"The current model has been already evaluated on the tasks: {executed_tasks}")
                playpen_eval_logger.info(f"Now attempting to evaluate on: {other_tasks}")
        else:
            for t in tasks:
                if t not in playpen_tasks:
                    raise ValueError("Task doesn't exist or is not a task in the Playpen Evaluation Pipeline.")

        playpen_eval_logger.info(f"Now evaluating on {tasks}")

        # Run evaluation for each task
        for task in tasks:
            backend = playpen_tasks[task]["backend"]
            if backend == "harness":
                try:
                    results = lm_eval.simple_evaluate(
                        model=model_backend,
                        model_args=model_args,
                        tasks=task,
                        device=device,
                        log_samples=True,
                        apply_chat_template=True,
                    )
                except:
                    results = lm_eval.simple_evaluate(
                        model=model_backend,
                        model_args=model_args,
                        tasks=task,
                        device=device,
                        log_samples=True,
                    )
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
                harness_results_file_path = Path(os.path.join(model_harness_results_path, f"{task}_harness_results_{timestamp}.json"))
                harness_results_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(harness_results_file_path, "w") as file:
                    json.dump(results, file, default=custom_json_serializer)
                playpen_results_file_path = Path(
                    os.path.join(model_playpen_results_path, f"{task}_playpen_results_{timestamp}.json"))
                playpen_results = prepare_playpen_results(main_task=task, model_name=model_name, harness_results=results)
                with open(playpen_results_file_path, "w") as file:
                    json.dump(playpen_results, file, default=custom_json_serializer)
            elif backend == "playpen":
                pass
                """results = _custom_evaluation(
                    model=model_backend,
                    model_args=model_args,
                    tasks=task,
                    device=device,
                    num_fewshot=0,
                    log_samples=True,
                )"""
            # we don't need to convert to playpen - we will do it automatically within the benchmarks

    def _custom_evaluation(self,
                           model,
                           model_args: Optional[Union[str, dict]] = None,
                           tasks: Optional[List[str]] = None,
                           num_fewshot: Optional[int] = None,
                           device: Optional[str] = None,
                           log_samples: bool = True,
                           ):
        pass

    @staticmethod
    def model_report(model_name: str, results_path:Path = "results") -> None:
        pass

    @staticmethod
    def benchmark_report(benchmark_name: str, results_path:Path = "results") -> None:
        pass

    @staticmethod
    def convert_res_from_harness(task_name: str, model_name:str, file_path: Path, output_path: Path) -> None:
        model_name = model_name.replace("/", "__")
        harness_dict = json.load(open(file_path, "r"))

        model_playpen_results_path = Path(os.path.join(project_folder, output_path)) / "playpen" / model_name
        model_playpen_results_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        playpen_results_file_path = Path(
            os.path.join(model_playpen_results_path, f"{task_name}_playpen_results_{timestamp}.json"))
        playpen_results = prepare_playpen_results(main_task=task_name,model_name=model_name, harness_results=harness_dict)
        with open(playpen_results_file_path, "w") as file:
            json.dump(playpen_results, file, default=custom_json_serializer)

