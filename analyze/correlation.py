import os
import abc
from abc import abstractmethod

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
from pathlib import Path

from config import project_root, get_capability_group_from_alias, get_alias, MODEL_REGISTRY
from analyze.score_extraction_utils import get_reports, get_scores, keep_common, organize_scores_capabilities, \
    sort_scores
from utils.utils import convert_str_to_number

class CorrelationMatrix(abc.ABC):
    @abstractmethod
    def sort(self):
        pass

class CorrelationMatrixBenchnmarks(CorrelationMatrix):

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

class CorrelationMatrixModels(CorrelationMatrix):
    def sort(self):
        pass


def sort_correlation_matrix(correlation_matrix: CorrelationMatrix):
    groups_info = []
    for task in correlation_matrix.data.columns:
        task_group = get_capability_group_from_alias(task)
        if task_group == "executive_functions":
            groups_info.append(0)
        elif task_group == "social_emotional_cognition":
            groups_info.append(1)
        elif task_group == "interactive":
            groups_info.append(2)
        elif task_group == "massive":
            groups_info.append(3)
        else:
            groups_info.append(4)
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
        0.5: 'red',
        1: 'blue',
        1.5: 'blue',
        2: 'green',
        3: 'purple'
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


def get_correlation_matrices_benchmarks(correlation_method:str, scores: Dict, lower_bound: float = None, upper_bound: float = None, partial: bool = False) -> List[CorrelationMatrix]:
    matrices = []
    for group, tasks in scores.items():
        if len(tasks.keys()) > 1:
            partial_scores = defaultdict(list)
            for task_id, model_results in tasks.items():
                task_name = get_alias(task_id)
                partial_scores[task_name] = [v for v in model_results if
                                    (lower_bound is None and upper_bound is None) or
                                    (lower_bound < convert_str_to_number(MODEL_REGISTRY[v[0]]['params']) <= upper_bound)]
            scores, remaining_models = keep_common(partial_scores)
            scores_matrix = pd.DataFrame(scores)
            correlation_matrix = CorrelationMatrixBenchnmarks(data=scores_matrix.corr(method=correlation_method), category=group, name=group, models=remaining_models)
            correlation_matrix = sort_correlation_matrix(correlation_matrix)
            matrices.append(correlation_matrix)
    return matrices

def get_correlation_matrices_models(correlation_method:str, scores: Dict, partial: bool = False) -> List[CorrelationMatrix]:
    task_names = sorted(set(t[0] for col in scores.values() for t in col))
    scores_matrix = pd.DataFrame(index=task_names)
    for model_name, tuples in scores.items():
        scores_matrix[model_name] = {t[0]: t[1] for t in tuples}
    matrices = []
    correlation_matrix = scores_matrix.corr(method=correlation_method)
    return matrices

"""def organize_scores_tasks(scores: Dict, task_registry: Dict):
    organized_scores = {
        "multiple_choice": defaultdict(lambda: defaultdict(list)),
        "open_question": defaultdict(lambda: defaultdict(list)),
        #"cloze": defaultdict(lambda: defaultdict(list)),
        "nli": defaultdict(lambda: defaultdict(list)),
        "minimal_pairs_logprobs": defaultdict(lambda: defaultdict(list)),
    }

    for task1, scores1 in scores.items():
        task1_type = task_registry[task1]["task_type"]
        for task2, scores2 in scores.items():
            task2_type = task_registry[task2]["task_type"]
            if (task1 != task2 and
                    len(task_registry[task1]["functional"]) > 0 and len(task_registry[task2]["functional"]) > 0 and isinstance(task1_type, str) and isinstance(task2_type, str) and
                    task1_type == task2_type):
                if len(task1_type) == 1 and len(task2_type) == 1:
                    organized_scores[task1_type[0]][task1] = scores1
    return organized_scores"""



def run_correlation(src_path: Path,
                    output_path_root:Path,
                    correlation_method: str,
                    discriminant: str,
                    partial: bool,
                    ignore_tasks: List[str],
                    ignore_groups: List[str],
                    tiers: bool,
                    benchmark_subset: str,
                    take_above_baseline: bool,
                    capability_groups_to_exclude: List[str],
                    by: str) -> None:
    src_path = project_root / src_path
    reports = get_reports(src_path=src_path)
    scores = get_scores(reports, benchmark_subset=benchmark_subset, ignore_tasks=ignore_tasks, ignore_groups=ignore_groups, take_above_baseline=take_above_baseline, by=by)
    sort_scores(scores, by=by)
    organized_scores = []
    if by == "models":
        organized_scores.append({"scores": scores,
                                 "output_path_root": output_path_root})
        correlation_matrices = get_correlation_matrices_models(correlation_method, organized_scores[0]["scores"])
        output_path = scores["output_path_root"]
        plot_and_save_matrices(correlation_matrices=correlation_matrices,
                               output_path_root=output_path)

    if by == "benchmarks":
        if discriminant == "capabilities":
            output_path_root = output_path_root/ "only_executive"
            organized_scores.append({"scores": organize_scores_capabilities(scores,
                                                                            capability_groups_to_exclude),
                                     "output_path_root": output_path_root}) # TODO: improve
            #elif discriminant == "tasks":
        #    organized_scores.append({"scores": organize_scores_tasks(scores, tasks_info), "output_path_root": output_path_root})
        elif discriminant == "benchmarks":
            organized_scores.append({"scores": scores, "output_path_root": output_path_root})
        for scores in organized_scores:

            correlation_matrices = get_correlation_matrices_benchmarks(correlation_method, scores["scores"], partial=partial)
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
                    correlation_matrices = get_correlation_matrices_benchmarks(correlation_method, scores["scores"], lower_bound=bounds['lower_bound'], upper_bound=bounds['upper_bound'])
                    output_path = scores["output_path_root"] / tier
                    plot_and_save_matrices(correlation_matrices=correlation_matrices,
                                           output_path_root=output_path)

