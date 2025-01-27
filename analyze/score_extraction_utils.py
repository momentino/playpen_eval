import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from config import get_task_info, get_model_registry
from utils.utils import convert_str_to_number

def get_reports(src_path: Path, model_registry: Dict[str, Dict[str, str]]):
    model_names = model_registry.keys()
    models_reports = []
    for subdir, dirs, files in os.walk(src_path):
        subfolder_name = os.path.basename(subdir)
        if subfolder_name in model_names or "tmp" in subfolder_name: #TODO TEMPORARY
            for file in files:
                if file.endswith('.json'):
                    models_reports.append(json.load(open(os.path.join(subdir, file))))
    return models_reports

def get_scores(reports, task_registry: Dict[str,Dict[str,Any]], ignore_tasks:List[str], ignore_groups: List[str], subset: str, take_above_baseline: bool):
    scores_dict = defaultdict(lambda: defaultdict(list))
    group_names = task_registry.keys()
    task_names = [task for g in group_names for task in task_registry[g].keys()]
    for report in reports:
        model_name = report["model_name"]
        for task_name, score_dict in report["task_results"].items():
            if task_name in task_names and task_name not in ignore_tasks:
                group, task_config = get_task_info(task_name)
                if group not in ignore_groups:
                    score = score_dict['score'] # score = score_dict['normalized_score']
                    if take_above_baseline:
                        if score < task_config["random_baseline"]:
                            continue
                    main_task = task_config["main_task"]
                    if subset == "subtasks" and (not main_task or len(task_registry[group]) == 1):
                        scores_dict[group][task_name].append((model_name, score))
                    elif subset == "main" and main_task:
                        scores_dict[group][task_name].append((model_name, score))
                    elif subset == "all":
                        scores_dict[group][task_name].append((model_name, score))

    return scores_dict

def sort_scores(scores: Dict):
    for group, tasks in scores.items():
        for task_name, model_scores in tasks.items():
            if len({t[0] for t in model_scores}) < len(model_scores):
                raise Exception(f"There are two scores for the same model and task! {[task_name]} Check your results files folder.")
            model_scores.sort(key=lambda x: (convert_str_to_number(get_model_registry()[x[0]]['params']), x[0]))

def organize_scores_capabilities(scores: Dict, task_registry: Dict, category:str):
    organized_scores = defaultdict(lambda: defaultdict(list))
    for group1, tasks1 in scores.items():
        for task1_name, scores1 in tasks1.items():
            for group2, tasks2 in scores.items():
                for task2_name, scores2 in tasks2.items():
                    if task1_name != task2_name:
                        if category != "total":
                            task1_categories = task_registry[group1][task1_name][category]
                            task2_categories = task_registry[group2][task2_name][category]
                            if len(task1_categories) == 1 and len(task2_categories) == 1 and task1_categories[0] == task2_categories[0]:
                                organized_scores[task2_categories[0]][task2_name] = scores2
                        else:
                            organized_scores["total"][task2_name] = scores2
    return organized_scores

def keep_common(partial_scores: Dict[str, List[Tuple[str,float]]]) -> (Dict[str, List[float]], set):
    sets_of_keys = [set(item[0] for item in lst) for lst in partial_scores.values()]
    common_keys = set.intersection(*sets_of_keys)
    remaining_models = set()
    filtered_data = {
        key: [tup[1] for tup in lst if tup[0] in common_keys and (remaining_models.add(tup[0]) or True)]
        for key, lst in partial_scores.items()
    }

    return filtered_data, remaining_models
