import os
import json
import pandas as pd
import frameworks.playeval_framework.evaluator as playeval
from eval import project_folder, lm_eval
from pathlib import Path
from typing import List
from datetime import datetime, timedelta
from eval import playpen_eval_logger, get_executed_tasks, get_playpen_tasks
from utils.utils import custom_json_serializer, convert_harness_results, compute_total_time

def list_tasks() -> None:
    # TODO
    pass

def get_task_backend(task: str, tasks_info: dict) -> str:
    for group, tasks in tasks_info.items():
        for name, info in tasks.items():
            if name == task:
                return info["backend"]
    return None

def run(model_backend: str, model_args: str, gen_kwargs:str, tasks: List, device: str, trust_remote_code:bool, parallelize:bool, results_path: Path = "results") -> None:

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
    if parallelize:
        model_args = model_args + ",parallelize=True"


    playpen_tasks = get_playpen_tasks()
    playpen_task_names = [name for task_info in playpen_tasks.values() for name in task_info.keys() if "main_task" in task_info[name].keys() and task_info[name]["main_task"]]
    if len(tasks) == 1:
        if "all" in tasks[0]:
            tasks = playpen_task_names
        elif "remaining" in tasks[0]:
            # Check for already executed tasks
            executed_tasks, other_tasks = get_executed_tasks(Path(model_playpen_results_path), playpen_task_names)
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
        start_time = datetime.now()
        backend = get_task_backend(task, playpen_tasks)
        assert backend is not None
        assert backend in {"harness", "playeval_framework"}
        if backend == "harness":
            results = lm_eval.simple_evaluate(
                model=model_backend,
                model_args=model_args,
                gen_kwargs=gen_kwargs,
                tasks=task,
                device=device,
                log_samples=True,
                apply_chat_template=True,
            )
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            harness_results_file_path = Path(os.path.join(model_harness_results_path, f"{task}_harness_results_{timestamp}.json"))
            harness_results_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(harness_results_file_path, "w") as file:
                json.dump(results, file, default=custom_json_serializer)
            results = convert_harness_results(model_name=model_name, harness_results=results)
        elif backend == "playeval_framework":
            results = playeval.evaluate(
                model=model_backend,
                model_args=model_args,
                gen_kwargs=gen_kwargs,
                task=task,
                device=device,
                log_samples=True,
                apply_chat_template=True,
            )
        end_time = datetime.now()
        task_time = end_time - start_time
        results["computing_time"] = str(task_time)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        playpen_results_file_path = Path(
            os.path.join(model_playpen_results_path, f"{task}_playpen_results_{timestamp}.json"))
        with open(playpen_results_file_path, "w") as file:
            json.dump(results, file, default=custom_json_serializer)

def report_costs(models: List[str], results_path: Path, output_path: Path) -> None:
    output_path = Path(os.path.join(project_folder, output_path))
    output_path.mkdir(parents=True, exist_ok=True)
    models =  [m.replace("/","__") for m in models]
    for m in models:
        results_path = Path(project_folder) / results_path / m
        report_name = m + ".csv"
        cost_report_path = output_path / report_name
        if cost_report_path.exists():
            try:
                df = pd.read_csv(cost_report_path)
            except Exception as e:
                print(f"Error reading the CSV file: {e}")
        else:
            # File does not exist, create a new CSV
            columns = ["benchmark", "date", "compute_time"]
            df = pd.DataFrame(columns=columns)

        for file_path in results_path.iterdir():
            if file_path.is_file() and file_path.suffix == '.json':
                file_name = file_path.stem
                benchmark_name, date = file_name.split("_playpen_results_")
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        computing_time = data.get("computing_time", "Key not found")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_path.name}")
                except Exception as e:
                    print(f"An error occurred with file {file_path.name}: {e}")

                if df.loc[df["benchmark"] == benchmark_name].empty:
                    new_row = {
                        "benchmark": benchmark_name,
                        "date": date,
                        "compute_time": computing_time
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    matching_row = df.loc[df["benchmark"] == benchmark_name, "date"]
                    date2 = matching_row.iloc[
                            0]
                    date_format = "%Y-%m-%dT%H-%M-%S.%f"
                    date_new = datetime.strptime(date, date_format)
                    date_old = datetime.strptime(date2, date_format)
                    if date_new > date_old:
                        # substitute with the more recent report
                        new_row = {
                            "benchmark": benchmark_name,
                            "date": date,
                            "compute_time": computing_time
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df = df[df['benchmark'] != 'TOTAL']
        total = compute_total_time(df['compute_time'].tolist())

        new_row = {
            "benchmark": "TOTAL",
            "date": datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f"),
            "compute_time": total
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        try:
            df.to_csv(cost_report_path, index=False)
        except Exception as e:
            print(f"Error creating the CSV file: {e}")



