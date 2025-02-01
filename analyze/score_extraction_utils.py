import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from config import get_task_info, MODEL_REGISTRY, TASK_REGISTRY
from utils.utils import convert_str_to_number

def get_reports(src_path: Path):
    model_names = MODEL_REGISTRY.keys()
    models_reports = []
    for subdir, dirs, files in os.walk(src_path):
        subfolder_name = os.path.basename(subdir)
        if subfolder_name in model_names or "bbh" in subfolder_name: #TODO TEMPORARY
            for file in files:
                if file.endswith('.json'):
                    models_reports.append(json.load(open(os.path.join(subdir, file))))
    return models_reports

def get_scores(reports, ignore_tasks:List[str], ignore_groups: List[str], benchmark_subset: str, take_above_baseline: bool, by:str):
    if by == "models":
        scores_dict = defaultdict(list)
    elif by == "benchmarks":
        scores_dict = defaultdict(lambda: defaultdict(list))
    group_names = TASK_REGISTRY.keys()
    task_names = [task for g in group_names for task in TASK_REGISTRY[g].keys()]
    for report in reports:
        model_name = report["model_name"]
        for task_name, score_dict in report["task_results"].items():
            if task_name in task_names and task_name not in ignore_tasks:
                group, task_config = get_task_info(task_name)
                if group not in ignore_groups:
                    score = score_dict['score'] # score = score_dict['normalized_score']
                    score = score/100 if score > 1 else score
                    if take_above_baseline:
                        if score < task_config["random_baseline"]:
                            continue
                    main_task = task_config["main_task"]
                    if by == "benchmarks":
                        if benchmark_subset == "subtasks" and (not main_task or len(TASK_REGISTRY[group]) == 1):
                            scores_dict[group][task_name].append((model_name, score))
                        elif benchmark_subset == "main" and main_task:
                            scores_dict[group][task_name].append((model_name, score))
                        elif benchmark_subset == "all":
                            scores_dict[group][task_name].append((model_name, score))
                    elif by == "models":
                        if benchmark_subset == "subtasks" and (not main_task or len(TASK_REGISTRY[group]) == 1):
                            scores_dict[model_name].append((task_name, score))
                        elif benchmark_subset == "main" and main_task:
                            scores_dict[model_name].append((task_name, score))
                        elif benchmark_subset == "all":
                            scores_dict[model_name].append((task_name, score))
    return scores_dict


def sort_scores(scores: Dict, by: str):
    if by=="benchmarks":
        for group, tasks in scores.items():
            for task_name, model_scores in tasks.items():
                if len({t[0] for t in model_scores}) < len(model_scores):
                    raise Exception(f"There are two scores for the same model and task! {[task_name]} Check your results files folder.")
                # sort according to family and within the family by the number of params
                model_scores.sort(key=lambda x: (MODEL_REGISTRY[x[0]]['family'],convert_str_to_number(MODEL_REGISTRY[x[0]]['params'])))
    elif by=="models":
        for model_name, model_scores in scores.items():
            if len({t[0] for t in model_scores}) < len(model_scores):
                raise Exception(
                    f"There are two scores for the same model and task! {[model_name]} Check your results files folder.")
            model_scores.sort(key=lambda x: (get_task_info(x[0])[1]["functional_group"], x[0])) # sort according to task name in alphabetical order

def organize_scores_capabilities(scores: Dict, capability_groups_to_exclude:List[str]):
    organized_scores = defaultdict(lambda: defaultdict(list))
    for group1, tasks1 in scores.items():
        for task1_name, scores1 in tasks1.items():
            for group2, tasks2 in scores.items():
                for task2_name, scores2 in tasks2.items():
                    if task1_name != task2_name:

                        task1_functional_group = TASK_REGISTRY[group1][task1_name]["functional_group"][0]
                        task2_functional_group = TASK_REGISTRY[group2][task2_name]["functional_group"][0]
                        if (task1_functional_group not in capability_groups_to_exclude) and (task2_functional_group not in capability_groups_to_exclude):
                            organized_scores[f"total_no_{capability_groups_to_exclude}"][task2_name] = scores2
                        """if category != "total":
                            task1_categories = task_registry[group1][task1_name][category]
                            task2_categories = task_registry[group2][task2_name][category]
                            if len(task1_categories) == 1 and len(task2_categories) == 1 and task1_categories[0] == task2_categories[0]:
                                organized_scores[task2_categories[0]][task2_name] = scores2
                        else:"""

    return organized_scores

def organize_scores_models(scores: Dict, task_registry: Dict, functional_groups_to_exclude:List[str]):
    organized_scores = defaultdict(lambda: defaultdict(list))

def keep_common(partial_scores: Dict[str, List[Tuple[str,float]]]) -> (Dict[str, List[float]], set):
    sets_of_keys = [set(item[0] for item in lst) for lst in partial_scores.values()]
    common_keys = set.intersection(*sets_of_keys)
    remaining_models = set()
    filtered_data = {
        key: [tup[1] for tup in lst if tup[0] in common_keys and (remaining_models.add(tup[0]) or True)]
        for key, lst in partial_scores.items()
    }

    return filtered_data, remaining_models
