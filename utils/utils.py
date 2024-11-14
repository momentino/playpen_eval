import os
from typing import List

import torch
from pathlib import Path
from datetime import datetime


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
    raise TypeError(f"Type {type(obj)} not serializable")

def get_all_json_files_from_path(dir: Path) -> List[str]:
    json_files_list = []
    for path in dir.rglob('*.json'):
        json_files_list.append(str(path.resolve()))
    return json_files_list

def prepare_playpen_results(task_name: str, model_name:str, harness_results: dict = None) -> dict:
    results = {}
    if(harness_results is not None):
        subtask_results = {}

        # TODO Improve, support other scores
        task_score_key = [key for key in harness_results["results"][task_name] if ("acc" in key or "f1" in key) and "stderr" not in key]

        task_score_key = task_score_key[0]
        aggregated_metric_name = task_score_key.split(",")[0]
        aggregated_score_value = harness_results["results"][task_name][task_score_key]
        aggregated_results = {"metric": aggregated_metric_name, "score": aggregated_score_value}

        results.update({
            "model_name": model_name,
            "task": task_name,
            "aggregated_results": aggregated_results,
            "subtask_results": subtask_results # TODO
        })
        return results
    raise Exception("Other options besides  harness are not yet implemented.")
