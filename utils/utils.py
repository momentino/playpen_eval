import os
from collections import defaultdict

import numpy as np
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
    elif isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def compute_fantom_aggregated_score(harness_results: dict) -> float:
    results = defaultdict(list)
    for task, samples in harness_results['samples'].items():
        if "fact" not in task:
            for sample in samples:
                set_id = sample['set_id']
                results[set_id].append(sample['acc'])

    all_ones_count = sum(1 for values in results.values() if all(score == 1 for score in values))
    num_set_id = len(results)

    all_star = all_ones_count / num_set_id
    return all_star

def prepare_playpen_results(main_task: str, model_name:str, harness_results: dict = None) -> dict:
    results = {}
    if(harness_results is not None):
        subtask_results = {}
        aggregated_results = {}

        for task_name, scores in harness_results["results"].items():

            # TODO Improve, support other scores
            task_score_key = [key for key in scores if ("acc" in key or "f1" in key or "exact_match" in key) and "stderr" not in key]

            task_score_key = task_score_key[0]
            metric_name = task_score_key.split(",")[0]
            score_value = scores[task_score_key]
            if task_name == "fantom_full":
                score_value = compute_fantom_aggregated_score(harness_results)
                aggregated_results = {"metric": 'all_star', "score": score_value}
            else:
                if(task_name == main_task):
                    aggregated_results = {"metric": metric_name, "score": score_value}
                else:
                    subtask_results[task_name] = {"metric": metric_name, "score": score_value}
        results.update({
            "model_name": model_name,
            "task": main_task,
            "aggregated_results": aggregated_results,
            "subtask_results": subtask_results
        })
        return results
    raise Exception("Other options besides  harness are not yet implemented.")
