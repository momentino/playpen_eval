import argparse
import json
import os
import logging
import lm_eval
from datetime import datetime
from pathlib import Path
from typing import List
from playpen_pipeline.utils import load_yaml_config
from lm_eval.tasks import TaskManager

pipeline_logger = logging.getLogger("pipeline-logger")


def get_executed_tasks(output_subfolder: Path, tasks: List[str]) -> (List[str], List[str]):
    executed_tasks = set()
    for json_file in output_subfolder.glob("*.json"):
        with open(json_file, "r") as file:
            try:
                data = json.load(file)
                if "results" in data:
                    executed_tasks.update(data["results"].keys())
            except json.JSONDecodeError:
                pipeline_logger.warning(f"Warning: {json_file} could not be decoded as JSON.")

    executed_tasks = list(executed_tasks)
    pending_tasks = [task for task in tasks if task not in executed_tasks]

    return executed_tasks, pending_tasks


def run_pipeline(args: argparse.Namespace) -> None:
    current_folder = Path(os.path.abspath(__file__)).parent
    project_folder = current_folder.parent
    config_file_path = Path(os.path.join(current_folder, "config.yaml"))
    config = load_yaml_config(config_file_path)

    tasks = config.get("tasks", [])
    model = config.get("model", "hf")
    device = config.get("device", "cuda:0")
    trust_remote_code = config.get("trust_remote_code", True)
    log_samples = config.get("log_samples", True)
    output_path = config.get("output_path", "playpen_results")

    model_args = args.model_args
    model_name = model_args.replace("pretrained=", "").replace("/", "__")
    if trust_remote_code:
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        args.model_args = args.model_args + ",trust_remote_code=True"

    output_subfolder = Path(os.path.join(project_folder, output_path)) / model_name
    output_subfolder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Check for already executed tasks
    executed_tasks, other_tasks = get_executed_tasks(output_subfolder, tasks)
    task_manager = TaskManager()
    harness_tasks = task_manager.all_tasks
    pending_tasks = [t for t in other_tasks if t in harness_tasks]
    unk_tasks = [t for t in other_tasks if t not in harness_tasks]

    pipeline_logger.info(f"The current model has been already evaluated on the tasks: {executed_tasks}")
    pipeline_logger.info(f"Now evaluating on {pending_tasks}")
    pipeline_logger.info(f"Unknown/Not yet implemented tasks: {unk_tasks}")

    # Run evaluation for each pending task
    for task in pending_tasks:
        results = lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=task,
            device=device,
            log_samples=log_samples,
        )
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        output_file_path = Path(os.path.join(output_subfolder,f"{task}_results{timestamp}.json"))
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w") as file:
            results = json.dumps(str(results))
            json.dump(results, file)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline.")
    parser.add_argument(
        "--model_args",
        type=str,
        required=True,
        help="Arguments for the model configuration, e.g., 'pretrained=model-name'."
    )

    args = parser.parse_args()
    run_pipeline(args)