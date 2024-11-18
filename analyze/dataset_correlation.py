import os
import json
import yaml
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
from pathlib import Path

from analyze import project_root
from itertools import chain, combinations
from utils.utils import convert_str_to_number

def get_models_reports(src_path: Path, models_names: List[str]):
    models_reports = []
    for subdir, dirs, files in os.walk(src_path):
        subfolder_name = os.path.basename(subdir)
        if subfolder_name in models_names:
            for file in files:
                if file.endswith('.json'):
                    models_reports.append(json.load(open(os.path.join(subdir, file))))
    return models_reports

def plot_and_save_matrix(correlation_matrix: pd.DataFrame, file_name:str, output_path: Path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
    file_path = output_path / file_name
    plt.title(correlation_matrix.name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

def build_correlation_matrices(scores_dict: Dict, save_results_path: Path) -> List[pd.DataFrame]:
    matrices = []
    # iterate over formal, functional, mixed
    for capability_type, subsets_dicts in scores_dict.items():
        # create a folder for each.
        capability_subset_type_path = Path(save_results_path) / capability_type
        if not os.path.exists(capability_subset_type_path):
            os.makedirs(capability_subset_type_path)

        # if there is more than one task sharing the same capabilities
        for capability_subset, task_dict in subsets_dicts.items():
            if len(task_dict.keys()) > 1:
                capability_subset_path = Path(capability_subset_type_path) / capability_subset
                if not os.path.exists(capability_subset_path):
                    os.mkdir(capability_subset_path)
                scores_by_task = defaultdict(list)
                for task, model_results in task_dict.items():
                    scores_by_task[task] = [v[2] for v in model_results ]

                scores_df = pd.DataFrame(scores_by_task)

                # Calculate the Pearson correlation matrix
                correlation_matrix = scores_df.corr()
                correlation_matrix.name = capability_subset
                matrices.append(correlation_matrix)

                plot_and_save_matrix(correlation_matrix=correlation_matrix, file_name=capability_subset, output_path=capability_subset_path)

def compute_correlation(scores_dict: Dict, task_info: Dict, capabilities_list: List,  output_path: Path) -> List[pd.DataFrame]:
    dict_for_correlations = {
        "functional": defaultdict(lambda: defaultdict(list)),
        "formal": defaultdict(lambda: defaultdict(list)),
        "mix": defaultdict(lambda: defaultdict(list)),
        "uncorrelated": defaultdict(lambda: defaultdict(list)),
        "total": defaultdict(lambda: defaultdict(list)),
    }

    for task1, scores1 in scores_dict.items():
        # list capabilities of task 1
        capabilities_task1 = copy.copy(task_info[task1]["functional"])
        capabilities_task1.extend(copy.copy(task_info[task1]["formal"]))

        for task2, scores2 in scores_dict.items():
            # list capabilities of task 2
            if(task1 != task2):
                common_capabilities = False
                capabilities_task2 = copy.copy(task_info[task2]["functional"])
                capabilities_task2.extend(copy.copy(task_info[task2]["formal"]))
                # get all subsets of capabilities in task 2
                all_possible_subsets_capabilities_task2 = list(
                    chain.from_iterable(combinations(capabilities_task2, r) for r in range(1, len(capabilities_task2) + 1)))
                # iterate over subsets
                for subset_capabilities_task2 in all_possible_subsets_capabilities_task2:
                    # if the subset is a subset of the capabilities of task 1, save the scores
                    if set(subset_capabilities_task2).issubset(set(capabilities_task1)):
                        common_capabilities = True
                        subset_key = ','.join(map(str, subset_capabilities_task2))
                        if set(subset_capabilities_task2).issubset(set(capabilities_list["functional"])):
                            capabilities_subset = "functional"
                        elif set(subset_capabilities_task2).issubset(set(capabilities_list["formal"])):
                            capabilities_subset = "formal"
                        else:
                            capabilities_subset = "mix"
                        dict_for_correlations[capabilities_subset]["total"][task2] = scores2
                        dict_for_correlations[capabilities_subset][subset_key][task2] = scores2
                if not common_capabilities:
                    # Build pairs of uncorrelated tasks
                    if len(dict_for_correlations["uncorrelated"][f"{task1}, {task2}"]) == 0 and len(dict_for_correlations["uncorrelated"][f"{task2}, {task1}"]) == 0:
                        dict_for_correlations["uncorrelated"][f"{task1}, {task2}"][task1] = scores1
                        dict_for_correlations["uncorrelated"][f"{task1}, {task2}"][task2] = scores2
    # create correlation matrices
    build_correlation_matrices(dict_for_correlations, output_path)

def run_correlation(src_path: Path, output_path:Path, take_subtasks: List[str]) -> None:
    available_models_path = Path(os.path.abspath(__file__)).parent.parent / "data" / "models_info_for_correlation.yaml"
    available_models_info = yaml.safe_load(open(available_models_path))["models"]
    available_models_names = available_models_info.keys()

    scores_dict = defaultdict(list)
    src_path = project_root / src_path
    models_reports = get_models_reports(src_path=src_path, models_names=available_models_names)

    # get file with listed the capabilities for each task
    task_info_path = Path(os.path.abspath(__file__)).parent.parent / "data" / "tasks_capabilities.yaml"
    task_info = yaml.safe_load(open(task_info_path))["capabilities_by_task"]
    capabilities_list = yaml.safe_load(open(task_info_path))["capabilities"]

    for report in models_reports:
        task_name = report["task"]
        model_name = report["model_name"]
        num_params = available_models_info[model_name]["params"]
        if task_name in take_subtasks:
            for subtask_name, subtask_results in report["subtask_results"].items():
                if subtask_name in task_info.keys():
                    score = subtask_results["score"]
                    results = (model_name, convert_str_to_number(num_params), score)
                    scores_dict[subtask_name].append(results)
        else:
            score = report["aggregated_results"]["score"]
            results = (model_name, convert_str_to_number(num_params), score)
            scores_dict[task_name].append(results)
    # Check for duplicates and sort by model name and param size
    for key, scores in scores_dict.items():
        if len({t[0] for t in scores}) < len(scores):
            raise Exception("There are two scores for the same model and task! Check your results files folder.")
        scores.sort(key=lambda x: (x[1], x[0]))


    compute_correlation(scores_dict, task_info, capabilities_list, output_path)
