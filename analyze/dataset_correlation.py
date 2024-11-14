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
from utils.utils import get_all_json_files_from_path, convert_str_to_number

def plot_and_save_matrix(correlation_matrix: pd.DataFrame, file_name:str, save_path: Path):
    # Create a heatmap with seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)

    # Save the figure as an image
    file_path = save_path / file_name
    print(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
    #plt.show()  # Display the heatmap

def build_correlation_matrices(dict_for_correlations: Dict, save_results_path: Path):
    # iterate over formal, functional, mixed

    for capability_subset_type, capability_subsets_dicts in dict_for_correlations.items():
        # create a folder for each.
        capability_subset_type_path = Path(save_results_path) / capability_subset_type
        if not os.path.exists(capability_subset_type_path):
            os.makedirs(capability_subset_type_path)

        dict_for_correlations_only_score = {}
        # if there is more than one task sharing the same capabilities
        #print(capability_subset_type)
        for capability_subset, task_dict in capability_subsets_dicts.items():
            #print(capability_subset)
            if len(task_dict.keys()) > 1:
                capability_subset_path = Path(capability_subset_type_path) / capability_subset
                if not os.path.exists(capability_subset_path):
                    os.mkdir(capability_subset_path)
                scores_by_task = defaultdict(list)
                for task, model_results in task_dict.items():
                    #print(model_results)
                    scores_by_task[task] = [v[2] for v in model_results ]

                df = pd.DataFrame(scores_by_task)

                # Calculate the correlation matrix
                correlation_matrix = df.corr()

                # display the correlation matrix
                print(f"Correlation matrix for {capability_subset}:")
                print(correlation_matrix)

                print("\n")
                file_name = capability_subset if capability_subset != "" else "total"
                plot_and_save_matrix(correlation_matrix, file_name, capability_subset_path)



def compute_correlation(results_grouped_by_task: Dict, task_info: Dict, full_capabilities_list: List,  save_results_path: Path) -> None:
    dict_for_correlations = {
        "functional": defaultdict(lambda: defaultdict(list)),
        "formal": defaultdict(lambda: defaultdict(list)),
        "mix": defaultdict(lambda: defaultdict(list))
    }

    for task1, scores1 in results_grouped_by_task.items():
        # list capabilities of task 1
        capabilities_task1 = copy.copy(task_info[task1]["functional"])
        capabilities_task1.extend(copy.copy(task_info[task1]["formal"]))

        for task2, scores2 in results_grouped_by_task.items():
            # list capabilities of task 2
            if(task1 != task2):
                capabilities_task2 = copy.copy(task_info[task2]["functional"])
                capabilities_task2.extend(copy.copy(task_info[task2]["formal"]))
                # get all subsets of capabilities in task 2
                all_possible_subsets_capabilities_task2 = list(
                    chain.from_iterable(combinations(capabilities_task2, r) for r in range(len(capabilities_task2) + 1)))

                # iterate over subsets
                for combination in all_possible_subsets_capabilities_task2:
                    # if the subset is a subset of the capabilities of task 1, save the scores
                    if set(combination).issubset(set(capabilities_task1)):
                        subset_key = ','.join(map(str, combination))
                        #print(combination)
                        if set(combination).issubset(set(full_capabilities_list["functional"])):
                            capabilities_subset = "functional"
                        elif set(combination).issubset(set(full_capabilities_list["formal"])):
                            capabilities_subset = "formal"
                        else:
                            capabilities_subset = "mix"

                        #print(subset_key, " ", task2, " ", scores2)
                        dict_for_correlations[capabilities_subset][subset_key][task2] = scores2


                        # create correlation plot (given subset, create plot for the two datasets)

                        # create report by benchmark. for each benchmark, build a report
                        # with which benchmarks did it correlate? Given each of the subsets of its skills,
                        # how many times, and with which benchmarks, did the dataset correlate
                        # with another among all those sharing the same subsets of skills?
                        # Count also how many are there for each subset.
        # create correlation matrices
    build_correlation_matrices(dict_for_correlations, save_results_path)


def run_correlation(data_path: str, save_results_path:str) -> None:
    model_info_path = Path(os.path.abspath(__file__)).parent.parent / "data" / "models_info_for_correlation.yaml"
    model_info = yaml.safe_load(open(model_info_path))["models"]

    results_grouped_by_task = {}
    data_path = project_root / data_path
    results_files_list = get_all_json_files_from_path(data_path)

    for file_path in results_files_list:
        results_data = json.load(open(file_path))
        task_name = results_data["task"]
        model_name = results_data["model_name"]
        num_params = model_info[model_name]["params"]
        score = results_data["aggregated_results"]["score"]
        results = (model_name, convert_str_to_number(num_params), score)
        results_grouped_by_task.setdefault(task_name, []).append(results)

    # Sort by model name and param size
    for key, value in results_grouped_by_task.items():
        value.sort(key=lambda x: (x[1], x[0]))
    # get file with listed the capabilities for each task
    task_info_path = Path(os.path.abspath(__file__)).parent.parent / "data" / "tasks_capabilities.yaml"
    task_info = yaml.safe_load(open(task_info_path))["capabilities_by_task"]
    capabilities_list = yaml.safe_load(open(task_info_path))["capabilities"]

    compute_correlation(results_grouped_by_task, task_info, capabilities_list, save_results_path)

