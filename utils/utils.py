import numpy as np
import torch
import json
import os
from typing import Dict
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from config import project_root, get_task_info

def prepare_reports_folders(framework: str, model_name: str, reports_path: str = 'results') -> Path:
    reports_path = Path(os.path.join(project_root, reports_path)) / framework / model_name
    reports_path.mkdir(parents=True, exist_ok=True)
    return reports_path

def convert_str_to_number(s: str) -> float:
    multipliers = {'B': 1_000_000_000}

    if s[-1] in multipliers:
        return float(s[:-1]) * multipliers[s[-1]]
    else:
        return float(s)

def custom_json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def compute_fantom_aggregated_score(harness_results: dict) -> float:
    results = defaultdict(list)
    for task, samples in harness_results['samples'].items():
        if "fact" not in task:
            for sample in samples:
                set_id = sample["doc"]["set_id"]
                if 'acc' in sample:
                    results[set_id].append(sample['acc'])
                elif 'f1' in sample:
                    results[set_id].append(sample['f1'])

    all_ones_count = sum(1 for values in results.values() if all(score == 1 for score in values))
    num_set_id = len(results)
    all_star = all_ones_count / num_set_id
    return all_star

def build_task_report(framework: str, task_name: str, model_name: str) -> Dict:
    if framework == 'clembench':
        return clembench_report(model_name, task_name)
    elif framework == 'lm-eval':
        return lmeval_report(model_name, task_name)

def clembench_report(model_name:str, game_name: str) -> dict:
    clembench_results_folder = project_root / "results" / "clembench" / model_name
    scores = []
    num_episodes = 0
    for file_path in clembench_results_folder.rglob("scores.json"):
        if file_path.is_file():
            if any(parent.name == game_name for parent in file_path.parents):
                try:
                    # Open and process the JSON file
                    with file_path.open('r') as file:
                        data = json.load(file)
                        if "episode scores" in data:
                            num_episodes+=1
                            main_score = data["episode scores"]["Main Score"]
                            if not np.isnan(main_score):
                                scores.append(data["episode scores"]["Main Score"])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    if len(scores) == 0:
        clemscore = 0
    else:
        played = len(scores) / num_episodes
        score = sum(scores)/len(scores)
        clemscore = score*(played/100)
    results = {"model_name":model_name, "task_results": {game_name:{"metric":"quality_score", "score":clemscore}}}
    return results

def lmeval_report(model_name:str, task_name: str):
    lmeval_full_report = project_root / "results" / "lm-eval" / model_name / f"{task_name}_report_latest.json"
    lmeval_report = json.load(open(lmeval_full_report))
    report = {}
    task_results = {}
    for task_name, scores in lmeval_report["results"].items():
        if task_name == "fantom_full":
            score_value = compute_fantom_aggregated_score(lmeval_report)
            task_results["fantom_full"] = {"metric": 'all_star', "score": score_value}
        else:
            _, task_info = get_task_info(task_name)
            metrics = task_info['metrics_short_report']
            task_results[task_name] = []
            for m in metrics:
                score = scores[m+',none']
                task_results[task_name].append({"metric": m, "score": score})
    report.update({
        "model_name": model_name,
        "task_results": task_results
    })
    return report

def time_to_seconds(time_str):
    h, m, s = time_str.split(":")
    s, ms = s.split(".") if "." in s else (s, "0")
    seconds = int(h) * 3600 + int(m) * 60 + int(s) + float(f"0.{ms}")
    return seconds



def compute_total_time(time_strings: str) -> str:
    total_seconds = sum(time_to_seconds(time) for time in time_strings)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02}:{seconds:06.3f}"


if __name__ == '__main__':
    from config import MODEL_REGISTRY
    for model_name, model_info in MODEL_REGISTRY.items():
        print(model_name)
        for game in ["wordle","imagegame","privateshared","referencegame","taboo"]:
            print(game)
            results = convert_clembench_results(model_name, game)
            print(results)
            results_path = "results"
            playpen_eval_results_path = Path(os.path.join(project_root, results_path)) / "playpen_eval" / model_name
            playpen_eval_results_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            playpen_results_file_path = Path(
            os.path.join(playpen_eval_results_path, f"{game}_fixed_playpen_results_{timestamp}.json"))
            with open(playpen_results_file_path, "w") as file:
                json.dump(results, file, default=custom_json_serializer)
