import os
import json
import datasets
from evaluate import lm_eval
from pathlib import Path
from typing import List
from datetime import datetime
from evaluate import get_logger, get_executed_tasks
from evaluate.normalize import normalize_scores
from config import project_root, get_task_registry, get_task_backend
from utils.utils import custom_json_serializer, convert_harness_results
import playpen.clemgame.benchmark as clembench_eval
from playpen.backends import read_model_specs
import frameworks.playpen_eval_benchmarks.evaluator as playeval

logger = get_logger(__name__)
stdout_logger = get_logger("evaluate.run")

def list_tasks() -> None:
    # TODO
    pass

def run(model_backend: str,
        model_args: str,
        gen_kwargs:str,
        tasks: List,
        device: str,
        trust_remote_code:bool,
        parallelize:bool,
        num_fewshot: int,
        fewshot_as_multiturn: bool,
        apply_chat_template: bool,
        results_path: Path = "results") -> None:

    model_name_parts = model_args.split(",")
    # Look for the part that starts with "pretrained="
    model_name = next(
        (part.replace("pretrained=", "").replace("/", "__") for part in model_name_parts if "pretrained=" in part),
        None  # Default value if "pretrained=" is not found
    )
    harness_results_path = Path(os.path.join(project_root, results_path)) / "harness" / model_name
    harness_results_path.mkdir(parents=True, exist_ok=True)

    clembench_results_path = Path(os.path.join(project_root, results_path)) / "clembench" / model_name
    clembench_results_path.mkdir(parents=True, exist_ok=True)

    playpen_eval_results_path = Path(os.path.join(project_root, results_path)) / "playpen_eval" / model_name
    playpen_eval_results_path.mkdir(parents=True, exist_ok=True)

    if trust_remote_code:
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        model_args = model_args + ",trust_remote_code=True"
    if parallelize:
        model_args = model_args + ",parallelize=True"

    task_registry = get_task_registry()
    task_names = [name for task_info in task_registry.values() for name in task_info.keys() if "main_task" in task_info[name].keys() and task_info[name]["main_task"]]
    if len(tasks) == 1:
        if "all" in tasks[0]:
            tasks = task_names
            stdout_logger.info(f"Now attempting to evaluate on all tasks available in the suite: {tasks}")
        elif "remaining" in tasks[0]:
            # Check for already executed tasks
            executed_tasks, other_tasks = get_executed_tasks(Path(playpen_eval_results_path), task_names)
            tasks = other_tasks
            stdout_logger.info(f"The current model has been already evaluated on the tasks: {executed_tasks}")
            stdout_logger.info(f"Now attempting to evaluate on: {other_tasks}")
    else:
        for t in tasks:
            if t not in task_names:
                message = f"Trying to evaluate on the requested tasks, but {t} is not available in the suite."
                stdout_logger.exception(message)
                raise ValueError(message)

    # Run evaluation for each task
    for task in tasks:
        start_time = datetime.now()
        backend = get_task_backend(task, task_registry)
        assert backend is not None
        assert backend in {"harness", "playpen_eval_benchmarks", "clembench"}
        if backend == "harness":
            results = lm_eval.simple_evaluate(
                model=model_backend,
                model_args=model_args,
                gen_kwargs=gen_kwargs,
                tasks=task,
                device=device,
                log_samples=True,
                num_fewshot=num_fewshot,
                fewshot_as_multiturn=fewshot_as_multiturn,
                apply_chat_template=apply_chat_template,
            )
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            harness_results_file_path = Path(os.path.join(harness_results_path, f"{task}_harness_results_{timestamp}.json"))
            harness_results_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(harness_results_file_path, "w") as file:
                json.dump(results, file, default=custom_json_serializer)
            results = convert_harness_results(model_name=model_name, harness_results=results)
        elif backend == "playpen_eval_benchmarks":
            results = playeval.evaluate(
                model=model_backend,
                model_args=model_args,
                gen_kwargs=gen_kwargs,
                task=task,
                device=device,
                log_samples=True,
                apply_chat_template=apply_chat_template,
            )
        elif backend == "clembench":
            clembench_eval.run(task,
                          model_specs=read_model_specs(model_name.split("__")[-1]),
                          gen_args=dict(pair.split("=") for pair in gen_kwargs.split(",")),
                          results_dir=str(clembench_results_path))
            clembench_eval.score(task, results_dir=str(clembench_results_path))
            clembench_eval.transcripts(task, results_dir=str(clembench_results_path))
        normalize_scores(results)
        end_time = datetime.now()
        task_time = end_time - start_time
        results["computing_time"] = str(task_time)
        stdout_logger.info(f"Evaluating {model_name} on {task} took {task_time}")
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        playpen_results_file_path = Path(
            os.path.join(playpen_eval_results_path, f"{task}_playpen_results_{timestamp}.json"))
        with open(playpen_results_file_path, "w") as file:
            json.dump(results, file, default=custom_json_serializer)

