import os
import json
import yaml
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set
from pathlib import Path

from analyze import project_root, model_registry_path, task_registry_path
from itertools import chain, combinations
from utils.utils import convert_str_to_number


class CorrelationMatrix():

    def __init__(self, data, capability_type, name):
        self.data = data
        self.capability_type = capability_type
        self.name = name


def get_model_registry():
    model_registry = yaml.safe_load(open(model_registry_path))["models"]
    return model_registry

def get_tasks_info():
    task_registry = yaml.safe_load(open(task_registry_path))
    tasks_info =  task_registry["tasks"]
    capabilities_list = task_registry["capabilities"]
    return capabilities_list, tasks_info


def get_reports(src_path: Path, model_registry: Dict[str, Dict[str, str]]):
    model_names = model_registry.keys()
    models_reports = []
    for subdir, dirs, files in os.walk(src_path):
        subfolder_name = os.path.basename(subdir)
        if subfolder_name in model_names:
            for file in files:
                if file.endswith('.json'):
                    models_reports.append(json.load(open(os.path.join(subdir, file))))
    return models_reports

def get_task_id(task_name: str, tasks_info: Dict[str, Dict[str, str]]):
    for key, item in tasks_info.items():
        if isinstance(item, dict) and item.get('alias') == task_name:
            return key
    return None

def sort_correlation_matrix(correlation_matrix: CorrelationMatrix, tasks_info: Dict[str, Dict[str, str]]):
    groups_info = []
    for task in correlation_matrix.data.columns:
        task_id = get_task_id(task, tasks_info)
        groups = tasks_info[task_id]["functional_groups"]
        if "executive_functions" in groups and "social_emotional_cognition" not in groups:
            groups_info.append(0)
        elif "executive_functions" in groups and "social_emotional_cognition" in groups:
            groups_info.append(1)
        elif "social_emotional_cognition" in groups and "executive_functions" not in groups:
            groups_info.append(2)
        else:
            groups_info.append(3)
    correlation_matrix.data["group"] = groups_info
    correlation_matrix_sorted = correlation_matrix.data.sort_values(by='group', axis=0)
    if not correlation_matrix_sorted.index.equals(correlation_matrix_sorted.columns):
        groups_info = correlation_matrix_sorted['group'].values.tolist()
        correlation_matrix_sorted = correlation_matrix_sorted.reindex(columns=correlation_matrix_sorted.index)
    correlation_matrix.data = correlation_matrix_sorted
    correlation_matrix.data["group"] = groups_info # TODO Improve. Reindex makes it disappear and so it is repeated twice
    return correlation_matrix


def plot_and_save_matrices(correlation_matrices: List[CorrelationMatrix], output_path_root: Path):
    color_map = {
        0: 'red',
        1: 'green',
        2: 'blue'
    }

    for matrix in correlation_matrices:
        output_path = output_path_root / matrix.capability_type
        os.makedirs(output_path, exist_ok=True)
        label_colors = [color_map[attr] for attr in matrix.data['group'].values.tolist()]
        matrix.data = matrix.data.drop(columns=['group'])
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            matrix.data.round(2),
            cmap='coolwarm',
            annot=True,  # Use the custom annotation matrix
            annot_kws={"size": 7},
            vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8}  # Optionally adjust color bar size
        )
        file_path = output_path / matrix.name
        # Color the labels (both x and y labels)
        for i, label in enumerate(matrix.data.columns):
            ax.get_xticklabels()[i].set_color(label_colors[i])
            ax.get_yticklabels()[i].set_color(label_colors[i])

        plt.savefig(f"{file_path}.png", dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
        plt.close()
        # save also dataframe
        matrix.data.to_csv(f"{file_path}.csv", index=True)

def get_correlation_matrices(correlation_method:str, scores: Dict, model_registry: Dict, tasks_info: Dict, lower_bound: float = None, upper_bound: float = None) -> List[CorrelationMatrix]:

    matrices = []
    # iterate over formal, functional, mixed
    for capability_type, subsets in scores.items():
        for subset_name, tasks in subsets.items():
            scores = defaultdict(list)
            for task, model_results in tasks.items():
                scores[task] = [v[1] for v in model_results if
                                (lower_bound is None and upper_bound is None) or
                                (lower_bound < convert_str_to_number(model_registry[v[0]]['params']) <= upper_bound)]
            scores_matrix = pd.DataFrame(scores)
            correlation_matrix = CorrelationMatrix(data=scores_matrix.corr(method=correlation_method), name=subset_name, capability_type=capability_type)
            correlation_matrix = sort_correlation_matrix(correlation_matrix, tasks_info)
            matrices.append(correlation_matrix)
    return matrices

def organize_scores(scores: Dict, tasks_info: Dict, capabilities_list: List[str]):
    organized_scores = {
        "functional": defaultdict(lambda: defaultdict(list)),
        "formal": defaultdict(lambda: defaultdict(list)),
        "mix": defaultdict(lambda: defaultdict(list)),
        "unrelated": defaultdict(lambda: defaultdict(list)),
        "total": defaultdict(lambda: defaultdict(list)),
    }

    for task1, scores1 in scores.items():
        task1_name = tasks_info[task1]["alias"]
        # list capabilities of task 1
        capabilities_task1 = copy.copy(tasks_info[task1]["functional"] + tasks_info[task1]["formal"])

        for task2, scores2 in scores.items():
            task2_name = tasks_info[task2]["alias"]
            # list capabilities of task 2
            if (task1 != task2):
                common_capabilities = False
                capabilities_task2 = copy.copy(tasks_info[task2]["functional"] + tasks_info[task2]["formal"])
                # get all subsets of capabilities in task 2
                task2_subsets = list(
                    chain.from_iterable(
                        combinations(capabilities_task2, r) for r in range(1, len(capabilities_task2) + 1)))
                # iterate over subsets
                for subset in task2_subsets:
                    subset_key = ','.join(map(str, subset))
                    if set(subset).issubset(set(capabilities_list["functional"])):
                        subset_type = "functional"
                    elif set(subset).issubset(set(capabilities_list["formal"])):
                        subset_type = "formal"
                    else:
                        subset_type = "mix"
                    # register overall correlation for all datasets associated with formal, functional capabilities or mixed
                    organized_scores[subset_type]["total"][task2_name] = scores2
                    # if the subset is a subset of the capabilities of task 1, save the scores
                    if set(subset).issubset(set(capabilities_task1)):
                        common_capabilities = True
                        organized_scores[subset_type][subset_key][task2_name] = scores2
                organized_scores["total"]["total"][task2_name] = scores2
                if not common_capabilities:
                    # Build pairs of uncorrelated tasks
                    if len(organized_scores["unrelated"][f"{task1_name}, {task2_name}"]) == 0 and len(
                            organized_scores["unrelated"][f"{task2_name}, {task1_name}"]) == 0:
                        organized_scores["unrelated"][f"{task1_name}, {task2_name}"][
                            task1_name] = scores1
                        organized_scores["unrelated"][f"{task1_name}, {task2_name}"][
                            task2_name] = scores2
    return organized_scores

def get_scores(reports, tasks_info: Dict[str,Dict[str,str]], tasks_to_ignore:List[str], tasks_to_unpack: List[str] = None):
    scores_dict = defaultdict(list)
    task_names = tasks_info.keys()
    for report in reports:
        task_name = report["task"]
        model_name = report["model_name"]
        if task_name not in tasks_to_ignore:
            if tasks_to_unpack is not None and task_name in tasks_to_unpack:
                for subtask_name, subtask_results in report["subtask_results"].items():
                    if subtask_name in task_names:
                        score = subtask_results["score"]
                        results = (model_name, score)
                        scores_dict[subtask_name].append(results)
            elif task_name in task_names:
                score = report["aggregated_results"]["score"]
                results = (model_name, score)
                scores_dict[task_name].append(results)
    return scores_dict

def run_correlation(src_path: Path, output_path_root:Path, correlation_method: str, tasks_to_ignore: List[str], tiers: bool, tasks_to_unpack:List[str] = None) -> None:

    model_registry = get_model_registry()
    capabilities_list, tasks_info = get_tasks_info()
    src_path = project_root / src_path
    reports = get_reports(src_path=src_path, model_registry = model_registry)
    scores = get_scores(reports, tasks_info, tasks_to_unpack=tasks_to_unpack, tasks_to_ignore=tasks_to_ignore)

    # Check for duplicates and sort by model name and param size
    for key, score_list in scores.items():
        if len({t[0] for t in score_list}) < len(score_list):
            raise Exception("There are two scores for the same model and task! Check your results files folder.")
        score_list.sort(key=lambda x: (convert_str_to_number(model_registry[x[0]]['params']), x[0]))

    organized_scores = organize_scores(scores, tasks_info, capabilities_list)

    correlation_matrices = get_correlation_matrices(correlation_method, organized_scores, model_registry = model_registry, tasks_info=tasks_info)
    output_path_root = output_path_root / "all"
    plot_and_save_matrices(correlation_matrices=correlation_matrices,
                           output_path_root=output_path_root)
    if tiers:
        model_size_thresholds = {
            'xsmall': {
                "lower_bound": 0,
                "upper_bound": 5000000000
            },
            'small': {
                "lower_bound": 5000000000,
                "upper_bound": 10000000000
            },
            #'medium': {
            #    "lower_bound": 10000000000,
            #    "upper_bound": 70000000000
            #}
        }
        for tier, bounds in model_size_thresholds.items():
            correlation_matrices = get_correlation_matrices(correlation_method, organized_scores, lower_bound=bounds['lower_bound'], upper_bound=bounds['upper_bound'], model_registry = model_registry, tasks_info=tasks_info)
            output_path_root = output_path_root / tier
            plot_and_save_matrices(correlation_matrices=correlation_matrices,
                                   output_path_root=output_path_root)



def verify_functional_correlation_patterns(src_path: Path):
    task_info_path = Path(os.path.abspath(__file__)).parent.parent / "data" / "tasks_details.yaml"
    task_info = yaml.safe_load(open(task_info_path))["tasks"]
    capabilities_list = yaml.safe_load(open(task_info_path))["capabilities"]
    pattern_root = src_path / "patterns"
    verification_pattern_root = pattern_root / "verification"



    # check for functional, and do it also at different model size tiers.
    for path in src_path.rglob('functional/total'):
        if path.is_dir():  # Ensure it's a directory
            for csv_file in path.glob('*.csv'):
                reorganized_correlations = defaultdict(list)
                correlation_df = pd.read_csv(csv_file, index_col=0)
                verification_output_path = verification_pattern_root / path.parent.parent.name # overall or model tier
                if not os.path.exists(verification_output_path):
                    os.makedirs(verification_output_path)

                # iterate over the matrix to check for patterns
                for task1 in correlation_df.columns:
                    # Find capabilities associated to column
                    task1_capabilities = []
                    for task_name, info in task_info.items():
                        if info["alias"] == task1:
                            task1_capabilities = info["functional"]

                    task2_capabilities = []
                    for task2, corr in correlation_df[task1].items():
                        if(task2 != task1):
                            # Find capabilities associated to row
                            for task_name, info in task_info.items():
                                if info["alias"] == task2:
                                    task2_capabilities = info["functional"]
                            num_common_capabilities = len(set(task1_capabilities).intersection(task2_capabilities))
                            reorganized_correlations[task1].append((num_common_capabilities, corr))

                # sort correlations for each skill so that the correlation with datasets which share most capabilities are first
                results = defaultdict(list)
                for task, correlations in reorganized_correlations.items():
                    reorganized_correlations[task] = sorted(correlations, key=lambda x: x[0], reverse=True)

                    # check whether it's true for each task that there is more correlation with tasks with which it shares more capabilities
                    if not all(t[0] == 0 for t in reorganized_correlations[task]): # if they have at least other capabilites in common with some other task
                        verified = True
                        for i, corr_tuple in enumerate(reorganized_correlations[task]):
                            if i>=0:
                                for j, corr_tuple2 in enumerate(reorganized_correlations[task][:i]):
                                    # if it has more capabilities in common
                                    if corr_tuple2[0] > corr_tuple[0]:
                                        if corr_tuple2[1] < corr_tuple[1]:
                                            verified = False
                                            break
                        results['task'].append(task)
                        results['verified'].append(verified)
                verify_df = pd.DataFrame(results)
                file_name = verification_output_path / "verify_results.csv"
                verify_df.to_csv(file_name, index=False)


                #print(reorganized_correlations)



