import os
import json
import datasets
import pandas as pd
from evaluate import lm_eval
from pathlib import Path
from typing import List
from datetime import datetime
from evaluate import get_logger, get_executed_tasks
from config import project_root, get_task_backend, TASK_REGISTRY, MAIN_TASK_LIST, COMPLETE_TASK_LIST
from utils.utils import custom_json_serializer, compute_total_time, prepare_reports_folders, build_task_report
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
        device_map_option:str,
        num_fewshot: int,
        fewshot_as_multiturn: bool,
        apply_chat_template: bool,
        batch_size: int) -> None:
    model_args_split = model_args.split(",")
    # Look for the part that starts with "pretrained=" or "peft="
    if "peft=" not in model_args:
        model_name = next(
            (part.replace("pretrained=", "").replace("/", "__") for part in model_args_split if "pretrained=" in part),
            None  # Default value if "pretrained=" is not found
        )
    else:
        model_name = next(
            (part.replace("peft=", "").replace("/", "__") for part in model_args_split if "peft=" in part),
            None ) # Default value if "peft=" is not found
    playpen_eval_reports_path = prepare_reports_folders('playpen_eval', model_name=model_name)

    if trust_remote_code:
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        model_args = model_args + ",trust_remote_code=True"
    if parallelize:
        model_args = model_args + ",parallelize=True"
    if device_map_option != "":
        model_args=model_args + f",device_map={device_map_option}"

    if len(tasks) == 1:
        if "all" in tasks[0]:
            tasks = MAIN_TASK_LIST
            stdout_logger.info(f"Now attempting to evaluate on all tasks available in the suite: {tasks}")
        elif "remaining" in tasks[0]:
            # Check for already executed tasks
            executed_tasks, other_tasks = get_executed_tasks(Path(playpen_eval_reports_path), MAIN_TASK_LIST)
            tasks = other_tasks
            stdout_logger.info(f"The current model has been already evaluated on the tasks: {executed_tasks}")
            stdout_logger.info(f"Now attempting to evaluate on: {other_tasks}")
    else:
        for t in tasks:
            if t not in COMPLETE_TASK_LIST:
                message = f"Trying to evaluate on the requested tasks, but {t} is not available in the suite."
                stdout_logger.exception(message)
                raise ValueError(message)
    # Run evaluation for each task
    for task in tasks:
        start_time = datetime.now()
        backend = get_task_backend(task)
        assert backend is not None
        assert backend in {"lm-eval", "playpen_eval_benchmarks", "clembench"}
        if backend == "lm-eval":
            lmeval_reports_path = prepare_reports_folders('lm-eval', model_name=model_name)
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
                batch_size=batch_size,
            )
            lmeval_report_path = Path(os.path.join(lmeval_reports_path, f"{task}_report_latest.json"))
            with open(lmeval_report_path, "w") as file:
                json.dump(results, file, default=custom_json_serializer)
        elif backend == "playpen_eval_benchmarks":
            report = playeval.evaluate(
                model=model_backend,
                model_args=model_args,
                gen_kwargs=gen_kwargs,
                task=task,
                device=device,
                log_samples=True,
                apply_chat_template=apply_chat_template,
            )
        elif backend == "clembench":
            clembench_reports_path = prepare_reports_folders('clembench', model_name=model_name)
            clembench_eval.run(task,
                          model_specs=read_model_specs([model_name.split("__")[-1]]),
                          gen_args = {'temperature': 0.0, 'max_tokens':250},
                          results_dir=str(clembench_reports_path))
            clembench_eval.score(task, results_dir=str(clembench_reports_path))
            clembench_eval.transcripts(task, results_dir=str(clembench_reports_path))

        report = build_task_report(backend, task, model_name) if backend in ["clembench", "lm-eval"] else report

        end_time = datetime.now()
        task_time = end_time - start_time
        report["computing_time"] = str(task_time)
        stdout_logger.info(f"Evaluating {model_name} on {task} took {task_time}")
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        playpen_results_file_path = Path(
            os.path.join(playpen_eval_reports_path, f"{task}_playpen_eval_report_{timestamp}.json"))
        with open(playpen_results_file_path, "w") as file:
            json.dump(report, file, default=custom_json_serializer)

def report_costs(models: List[str], results_path: Path, output_path: Path) -> None:
    output_path = Path(os.path.join(project_root, output_path))
    output_path.mkdir(parents=True, exist_ok=True)
    models =  [m.replace("/","__") for m in models]
    for m in models:
        results_path = Path(project_root) / results_path / m
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
