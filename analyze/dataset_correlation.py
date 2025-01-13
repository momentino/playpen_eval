import os
import json
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from pathlib import Path

from analyze import project_root, model_registry_path, task_registry_path
from utils.utils import convert_str_to_number


class CorrelationMatrix():

    def __init__(self, data, category, name, models):
        self.data = data
        self.category = category
        self.name = name
        self.models = models

    def sort(self) -> None:
        self.data = self.data.sort_values(by='group')
        self.data = self.data.groupby('group', group_keys=False).apply(
            lambda group: group.sort_index()
        )

        # for having the same sorting on rows and columns
        group_column = self.data.pop('group')  # Remove 'group' column
        self.data = self.data[self.data.index]
        self.data['group'] = group_column


def get_model_registry():
    model_registry = yaml.safe_load(open(model_registry_path))["models"]
    return model_registry

def get_tasks_info():
    task_registry = yaml.safe_load(open(task_registry_path))
    tasks_info =  task_registry["groups"]
    capabilities_list = task_registry["capabilities"]
    return capabilities_list, tasks_info

def get_alias(task_name:str, tasks_info: Dict) -> str:
    for group, tasks in tasks_info.items():
        for name, info in tasks.items():
            if name == task_name:
                return info["alias"]

def get_functional_group_from_alias(task_name:str, tasks_info: Dict) -> str:
    for group, tasks in tasks_info.items():
        for name, info in tasks.items():
            if info["alias"] == task_name:
                return info["functional_groups"]

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

def sort_correlation_matrix(correlation_matrix: CorrelationMatrix, tasks_info: Dict[str, Dict[str, str]]):
    groups_info = []
    for task in correlation_matrix.data.columns:
        functional_groups = get_functional_group_from_alias(task, tasks_info)
        if "executive_functions" in functional_groups and "social_emotional_cognition" not in functional_groups:
            groups_info.append(0)
        elif "executive_functions" in functional_groups and "social_emotional_cognition" in functional_groups:
            groups_info.append(1)
        elif "social_emotional_cognition" in functional_groups and "executive_functions" not in functional_groups:
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
        matrix.sort()
        matrix_data = matrix.data
        output_path = output_path_root / matrix.category
        os.makedirs(output_path, exist_ok=True)
        label_colors = [color_map[attr] for attr in matrix_data['group'].values.tolist()]
        matrix_data = matrix_data.drop(columns=['group'])

        num_rows, num_cols = matrix_data.shape
        cell_width = 1.0
        cell_height = 0.6
        figsize = (num_cols * cell_width, num_rows * cell_height)

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            matrix_data.round(2),
            cmap='coolwarm',
            annot=True,  # Use the custom annotation matrix
            annot_kws={"size": 7},
            vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8}  # Optionally adjust color bar size
        )
        file_path = output_path / matrix.name
        # Color the labels (both x and y labels)
        for i, label in enumerate(matrix_data.columns):
            ax.get_xticklabels()[i].set_color(label_colors[i])
            ax.get_yticklabels()[i].set_color(label_colors[i])

        plt.savefig(f"{file_path}.png", dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
        plt.close()
        # save also dataframe
        matrix_data.to_csv(f"{file_path}.csv", index=True)
        # save models taken into consideration
        file_path = output_path / f"{matrix.name}_models.txt"
        with open(file_path, "w") as file:
            for element in matrix.models:
                file.write(f"{element}\n")


def keep_common(partial_scores: Dict[str, List[Tuple[str,float]]]) -> (Dict[str, List[float]], set):
    sets_of_keys = [set(item[0] for item in lst) for lst in partial_scores.values()]
    common_keys = set.intersection(*sets_of_keys)
    remaining_models = set()
    filtered_data = {
        key: [tup[1] for tup in lst if tup[0] in common_keys and (remaining_models.add(tup[0]) or True)]
        for key, lst in partial_scores.items()
    }

    return filtered_data, remaining_models

def get_correlation_matrices(correlation_method:str, scores: Dict, model_registry: Dict, tasks_info: Dict, lower_bound: float = None, upper_bound: float = None) -> List[CorrelationMatrix]:
    matrices = []
    for category, tasks in scores.items():
        if len(tasks.keys()) > 1:
            partial_scores = defaultdict(list)
            for task_id, model_results in tasks.items():
                task_name = get_alias(task_id, tasks_info)
                partial_scores[task_name] = [v for v in model_results if
                                    (lower_bound is None and upper_bound is None) or
                                    (lower_bound < convert_str_to_number(model_registry[v[0]]['params']) <= upper_bound)]
            scores, remaining_models = keep_common(partial_scores)
            scores_matrix = pd.DataFrame(scores)
            correlation_matrix = CorrelationMatrix(data=scores_matrix.corr(method=correlation_method), category=category, name=category, models=remaining_models)
            correlation_matrix = sort_correlation_matrix(correlation_matrix, tasks_info)
            matrices.append(correlation_matrix)
    return matrices

def organize_scores_capabilities(scores: Dict, tasks_info: Dict, category:str):
    organized_scores = defaultdict(lambda: defaultdict(list))
    for group1, tasks1 in scores.items():
        for task1_name, scores1 in tasks1.items():
            for group2, tasks2 in scores.items():
                for task2_name, scores2 in tasks2.items():
                    if (task1_name != task2_name):
                        if category != "total":
                            task1_categories = tasks_info[group1][task1_name][category]
                            task2_categories = tasks_info[group2][task2_name][category]
                            if len(task1_categories) == 1 and len(task2_categories) == 1 and task1_categories[0] == task2_categories[0]:
                                organized_scores[task2_categories[0]][task2_name] = scores2
                        else:
                            organized_scores["total"][task2_name] = scores2
    return organized_scores

def organize_scores_tasks(scores: Dict, tasks_info: Dict):
    organized_scores = {
        "multiple_choice": defaultdict(lambda: defaultdict(list)),
        "open_question": defaultdict(lambda: defaultdict(list)),
        #"cloze": defaultdict(lambda: defaultdict(list)),
        "nli": defaultdict(lambda: defaultdict(list)),
        "minimal_pairs_logprobs": defaultdict(lambda: defaultdict(list)),
    }

    for task1, scores1 in scores.items():
        task1_type = tasks_info[task1]["task_type"]
        for task2, scores2 in scores.items():
            task2_type = tasks_info[task2]["task_type"]
            if (task1 != task2 and
                    len(tasks_info[task1]["functional"]) > 0 and len(tasks_info[task2]["functional"]) > 0 and isinstance(task1_type, str) and isinstance(task2_type, str) and
                    task1_type == task2_type):
                if len(task1_type) == 1 and len(task2_type) == 1:
                    organized_scores[task1_type[0]][task1] = scores1
    return organized_scores

"""def organize_scores_benchmarks(scores: Dict, tasks_info: Dict):
    organized_scores = defaultdict(lambda: defaultdict(list))

    for group1, group1_results in scores.items():
        for task1, scores1 in group1_results.items():
            for group2, group2_results in scores.items():
                for task2, scores2 in group2_results.items():
                    if (task1 != task2 and
                            len(tasks_info[task1]["functional"]) > 0 and len(tasks_info[task2]["functional"]) > 0 and
                            group1 == group2):
                            organized_scores[group1][task1] = scores1
    return organized_scores"""

def ammissible(score: float, chance_level:float) -> bool:
    if score > chance_level:
        return True
    return False

def get_info(task_name: str, tasks_info: Dict) -> (str, float, bool, bool):
    for group, tasks in tasks_info.items():
        for name, info in tasks.items():
            if name == task_name:
                functional = len(info["functional"])>0 and len(info["formal"])==0
                try:
                    main_task = info["main_task"]
                except:
                    main_task = False
                return group, info["random_baseline"], main_task, functional
    return None, None, None, None

def get_scores(reports, tasks_info: Dict[str,Dict[str,Any]], tasks_to_ignore:List[str], take_functional_subtasks: bool):
    scores_dict = defaultdict(lambda: defaultdict(list))
    group_names = tasks_info.keys()
    task_names = [task for g in group_names for task in tasks_info[g].keys()]

    for report in reports:
        model_name = report["model_name"]
        for task_name, score_dict in report["task_results"].items():
            score = score_dict['score']
            if task_name in task_names and task_name not in tasks_to_ignore:
                group_name, chance_level, main_task, functional = get_info(task_name, tasks_info)
                assert group_name is not None
                assert chance_level is not None
                assert main_task is not None
                assert functional is not None
                if take_functional_subtasks and functional and ammissible(score, chance_level) and not main_task:
                    print(task_name)
                    scores_dict[group_name][task_name].append((model_name, score))
                elif not take_functional_subtasks and functional and main_task and ammissible(score, chance_level):
                    scores_dict[group_name][task_name].append((model_name, score))
    return scores_dict

def run_correlation(src_path: Path, output_path_root:Path, correlation_method: str, discriminant: str, tasks_to_ignore: List[str], tiers: bool, take_functional_subtasks: bool) -> None:

    model_registry = get_model_registry()
    capabilities_list, tasks_info = get_tasks_info()
    src_path = project_root / src_path
    reports = get_reports(src_path=src_path, model_registry = model_registry)
    scores = get_scores(reports, tasks_info, take_functional_subtasks=take_functional_subtasks, tasks_to_ignore=tasks_to_ignore)

    # Check for duplicates and sort by model name and param size
    for group, tasks in scores.items():
        for task_name, model_scores in tasks.items():
            if len({t[0] for t in model_scores}) < len(model_scores):
                raise Exception("There are two scores for the same model and task! Check your results files folder.")
            model_scores.sort(key=lambda x: (convert_str_to_number(model_registry[x[0]]['params']), x[0]))

    organized_scores = []
    if discriminant == "capabilities":
        output_path_root = output_path_root/ "functional"
        organized_scores.append({"scores": organize_scores_capabilities(scores, tasks_info, "functional"), "output_path_root": output_path_root}) # TODO: improve
        output_path_root = output_path_root / "total"
        organized_scores.append(
            {"scores": organize_scores_capabilities(scores, tasks_info, "total"),
             "output_path_root": output_path_root})  # TODO: improve

    elif discriminant == "tasks":
        organized_scores.append({"scores": organize_scores_tasks(scores, tasks_info), "output_path_root": output_path_root})
    elif discriminant == "benchmarks":
        organized_scores.append({"scores": scores, "output_path_root": output_path_root})


    for scores in organized_scores:
        correlation_matrices = get_correlation_matrices(correlation_method, scores["scores"], model_registry = model_registry, tasks_info=tasks_info)
        output_path_root = output_path_root / "all"
        plot_and_save_matrices(correlation_matrices=correlation_matrices,
                               output_path_root=scores["output_path_root"])
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
                correlation_matrices = get_correlation_matrices(correlation_method, scores["scores"], lower_bound=bounds['lower_bound'], upper_bound=bounds['upper_bound'], model_registry = model_registry, tasks_info=tasks_info)
                output_path = scores["output_path_root"] / tier
                plot_and_save_matrices(correlation_matrices=correlation_matrices,
                                       output_path_root=output_path)



