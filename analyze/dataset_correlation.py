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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

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

def sort_labels(correlation_matrix: pd.DataFrame, scores_matrix: pd.DataFrame):
    dissimilarity = 1 - correlation_matrix
    dissimilarity = np.clip(dissimilarity, 0.0, None)
    Z = linkage(squareform(dissimilarity), 'average')
    # Clusterize the data
    threshold = 0.2
    labels = fcluster(Z, threshold, criterion='distance')

    # Keep the indices to sort labels
    ordered_labels = np.argsort(labels)
    return ordered_labels

def cluster(ordered_labels: np.ndarray, scores_matrix: pd.DataFrame):

    # Build a new dataframe with the sorted columns
    for idx, i in enumerate(scores_matrix.columns[ordered_labels]):
        if idx == 0:
            clustered = pd.DataFrame(scores_matrix[i])
        else:
            df_to_append = pd.DataFrame(scores_matrix[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)

    return clustered


def plot_and_save_matrix(correlation_matrix: pd.DataFrame, name: bool, file_name:str, output_path: Path):
    plt.figure(figsize=(8, 6))

    #mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    ax = sns.heatmap(
        correlation_matrix.round(2),
        #mask=mask,
        cmap='coolwarm',
        annot=True,  # Use the custom annotation matrix
        annot_kws={"size": 7},
        vmin=-1, vmax=1,
        cbar_kws={'shrink': 0.8}  # Optionally adjust color bar size
    )

    # Hide the diagonal ticks
    #ax.set_xticks(np.delete(ax.get_xticks(), len(correlation_matrix) - 1))
    #ax.set_yticks(np.delete(ax.get_yticks(), 0))
    file_path = output_path / file_name

    if name:
        plt.title(correlation_matrix.name)
    plt.savefig(f"{file_path}.png", dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
    plt.close()
    # save also dataframe
    correlation_matrix.to_csv(f"{file_path}.csv", index=True)

def build_correlation_matrices(scores_dict: Dict, divide_by_model_size:bool, name:bool, save_results_path: Path) -> List[pd.DataFrame]:
    model_size_thresholds = {
        'xsmall':{
            "lower_bound":0,
            "upper_bound": 5000000000
        },
        'small': {
            "lower_bound": 5000000000,
            "upper_bound": 10000000000
        },
        'medium': {
            "lower_bound": 10000000000,
            "upper_bound": 70000000000
        }
    }

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
                output_path = Path(capability_subset_type_path) / capability_subset
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                scores_by_task = defaultdict(list)
                for task, model_results in task_dict.items():
                    scores_by_task[task] = [v[2] for v in model_results ]
                scores_matrix = pd.DataFrame(scores_by_task)
                # Calculate the Pearson correlation matrix
                correlation_matrix = scores_matrix.corr()
                correlation_matrix.name = capability_subset
                matrices.append(correlation_matrix)

                ordered_labels = sort_labels(correlation_matrix, scores_matrix)
                clustered = cluster(ordered_labels, scores_matrix)
                clustered_corr = clustered.corr()
                clustered_corr.name = capability_subset
                plot_and_save_matrix(correlation_matrix=clustered_corr, name=name, file_name=capability_subset, output_path=output_path)

                # Calculate matrices for models in specific tiers
                if divide_by_model_size:
                    for i,(tier, bounds) in enumerate(model_size_thresholds.items()):
                        for task, model_results in task_dict.items():
                            scores_by_task[task] = [v[2] for v in model_results if v[1] > bounds["lower_bound"] and v[1] <= bounds["upper_bound"] ]

                        scores_matrix = pd.DataFrame(scores_by_task)

                        # check if there are at least some results for this tier
                        if not scores_matrix.isna().values.all():
                            correlation_matrix = scores_matrix.corr()

                            matrices.append(correlation_matrix)
                            output_path = Path(save_results_path) / tier / capability_type / capability_subset
                            if not os.path.exists(output_path):
                                os.makedirs(output_path)
                            # We don't do it again so we can have the same configuration as the overall matrices also for those divided by tiers
                            # And have an easier time comparing results
                            clustered = cluster(ordered_labels, scores_matrix)
                            clustered_corr = clustered.corr()
                            clustered_corr.name = capability_subset
                            plot_and_save_matrix(correlation_matrix=clustered_corr, name=name, file_name=capability_subset,
                                                 output_path=output_path)

def compute_correlation(scores_dict: Dict, task_info: Dict, capabilities_list: List, divide_by_model_size:bool, output_path: Path, name:bool) -> List[pd.DataFrame]:
    formatted_scores_dict = {
        "functional": defaultdict(lambda: defaultdict(list)),
        "formal": defaultdict(lambda: defaultdict(list)),
        "mix": defaultdict(lambda: defaultdict(list)),
        "unrelated": defaultdict(lambda: defaultdict(list)),
        "total": defaultdict(lambda: defaultdict(list)),
    }

    for task1, scores1 in scores_dict.items():
        task1_prettyname = task_info[task1]["alias"]
        # list capabilities of task 1
        capabilities_task1 = copy.copy(task_info[task1]["functional"])
        capabilities_task1.extend(copy.copy(task_info[task1]["formal"]))

        for task2, scores2 in scores_dict.items():
            task2_prettyname = task_info[task2]["alias"]
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
                    subset_key = ','.join(map(str, subset_capabilities_task2))
                    if set(subset_capabilities_task2).issubset(set(capabilities_list["functional"])):
                        capabilities_subset = "functional"
                    elif set(subset_capabilities_task2).issubset(set(capabilities_list["formal"])):
                        capabilities_subset = "formal"
                    else:
                        capabilities_subset = "mix"
                    # register overall correlation for all datasets associated with formal, functional capabilities or mixed
                    formatted_scores_dict[capabilities_subset]["total"][task2_prettyname] = scores2
                    # if the subset is a subset of the capabilities of task 1, save the scores
                    if set(subset_capabilities_task2).issubset(set(capabilities_task1)):
                        common_capabilities = True
                        formatted_scores_dict[capabilities_subset][subset_key][task2_prettyname] = scores2
                formatted_scores_dict["total"]["total"][task2_prettyname] = scores2
                if not common_capabilities:
                    # Build pairs of uncorrelated tasks
                    if len(formatted_scores_dict["unrelated"][f"{task1_prettyname}, {task2_prettyname}"]) == 0 and len(formatted_scores_dict["unrelated"][f"{task2_prettyname}, {task1_prettyname}"]) == 0:
                        formatted_scores_dict["unrelated"][f"{task1_prettyname}, {task2_prettyname}"][task1_prettyname] = scores1
                        formatted_scores_dict["unrelated"][f"{task1_prettyname}, {task2_prettyname}"][task2_prettyname] = scores2
    # create correlation matrices
    build_correlation_matrices(formatted_scores_dict, divide_by_model_size, name, output_path)

def run_correlation(src_path: Path, output_path:Path, take_subtasks: List[str], tasks_to_ignore: List[str], divide_by_model_size: bool, name:bool) -> None:
    if len(take_subtasks) > 0:
        output_path =output_path / Path(f"subt_ver_{','.join(take_subtasks)}")
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = output_path / Path("overall")

    available_models_path = Path(os.path.abspath(__file__)).parent.parent / "data" / "models_info_for_correlation.yaml"
    available_models_info = yaml.safe_load(open(available_models_path))["models"]
    available_models_names = available_models_info.keys()

    scores_dict = defaultdict(list)
    src_path = project_root / src_path
    models_reports = get_models_reports(src_path=src_path, models_names=available_models_names)

    # get file with listed the capabilities for each task
    task_info_path = Path(os.path.abspath(__file__)).parent.parent / "data" / "tasks_details.yaml"
    task_info = yaml.safe_load(open(task_info_path))["tasks"]
    capabilities_list = yaml.safe_load(open(task_info_path))["capabilities"]

    for report in models_reports:
        task_name = report["task"]
        model_name = report["model_name"]
        num_params = available_models_info[model_name]["params"]
        if task_name not in tasks_to_ignore:
            if task_name in take_subtasks:
                for subtask_name, subtask_results in report["subtask_results"].items():
                    if subtask_name in task_info.keys():
                        score = subtask_results["score"]
                        results = (model_name, convert_str_to_number(num_params), score)
                        scores_dict[subtask_name].append(results)
            elif task_name in task_info.keys():
                score = report["aggregated_results"]["score"]
                results = (model_name, convert_str_to_number(num_params), score)
                scores_dict[task_name].append(results)
    # Check for duplicates and sort by model name and param size
    for key, scores in scores_dict.items():
        if len({t[0] for t in scores}) < len(scores):
            raise Exception("There are two scores for the same model and task! Check your results files folder.")
        scores.sort(key=lambda x: (x[1], x[0]))
    compute_correlation(scores_dict, task_info, capabilities_list, divide_by_model_size, output_path, name)

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



