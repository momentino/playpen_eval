import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib.cm import get_cmap
from pathlib import Path
from typing import Dict, List
from utils.utils import convert_str_to_number
from config import project_root, get_alias, get_baseline, get_task_info, MODEL_REGISTRY
from analyze.score_extraction_utils import get_reports, get_scores, sort_scores
from scipy.stats import rankdata

benchmark_magnitudes = {
    "cladder": 0.2,
    "llm_cognitive_flexibility_lnt": 0.4,
    "llm_cognitive_flexibility_wcst": 0.5,
    "logiqa2": 0.6,
    "planbench_blocksworld_plan_generation": 0.7,
    "winogrande": 0.8,
    "wm_verbal_3back": 1.0,
    "behavior-qa":0.1,
    "eq_bench": 0.2,
    "lm_pragmatics": 0.4,
    "judgment-qa": 0.6,
    "mental-state-qa":0.8,
    "social_iqa": 1.0,
    "wordle":0.1,
    "privateshared": 0.2,
    "imagegame": 0.5,
    "referencegame": 0.4,
    "taboo": 0.6,
    "bbh_fewshot": 0.6,
    "ewok_minimal_pairs": 0.9,
    "mmlu": 0.3,

}

def choose_color_benchmarks(benchmark_name):
    _, info = get_task_info(benchmark_name)
    functional_group = info["functional_group"][0]
    if functional_group == "executive_functions":
        return get_cmap("Reds")(benchmark_magnitudes[benchmark_name])
    elif functional_group == "social_emotional_cognition":
        return get_cmap("Blues")(benchmark_magnitudes[benchmark_name])
    elif functional_group == "massive":
        return get_cmap("Greens")(benchmark_magnitudes[benchmark_name])
    elif functional_group == "interactive":
        return get_cmap("Purples")(benchmark_magnitudes[benchmark_name])
def choose_color_models(model_name, magnitude):
    """Select colormap based on s[1] value range."""
    if "Qwen" in model_name:
        return get_cmap("Purples")(magnitude)  # Lighter shades for lower values
    elif "Llama" in model_name:
        return get_cmap("Greens")(magnitude)  # Mid-range values get green
    elif "OLMo" in model_name:
        return get_cmap("Reds")(magnitude)  # Higher values get red shades
    elif "Falcon" in model_name:
        return get_cmap("Greys")(magnitude)



def build_and_save_scatterplots_benchmarks(scores: Dict, output_path_root: Path, correlation_matrix: pd.DataFrame, p_values_matrix: pd.DataFrame, correlation_method: str):
    total_iterations = sum(len(tasks1) * len(tasks2) for tasks1 in scores.values() for tasks2 in scores.values())
    with tqdm(total=total_iterations, desc="Building scatterplots", unit="scatterplot") as pbar:
        for group1, tasks1 in scores.items():
            for task1_name, scores1 in tasks1.items():
                for group2, tasks2 in scores.items():
                    for task2_name, scores2 in tasks2.items():
                        #pair = tuple(sorted([task1_name, task2_name]))
                        #if task1_name != task2_name and pair not in processed_pairs:
                        #    processed_pairs.add(pair)
                        task1_alias = get_alias(task1_name)
                        task2_alias = get_alias(task2_name)

                        x = scores1
                        y = scores2

                        x = [(s[0],s[1] / 100) if s[1] > 1 else (s[0],s[1]) for s in x]
                        y = [(s[0],s[1] / 100) if s[1] > 1 else (s[0],s[1]) for s in y]

                        x_labels, x_values = zip(*x)
                        y_labels, y_values = zip(*y)
                        # Ensure labels match between x and y
                        assert x_labels == y_labels, f"Labels in x {x_labels} and y {y_labels} for the tasks {task1_name},{task2_name} do not match!"

                        # Assume model_size_dict is a dictionary mapping labels to model sizes
                        model_sizes = [convert_str_to_number(MODEL_REGISTRY[label]['params']) for label in x_labels]  # Extract model sizes
                        size_ranks = rankdata(model_sizes, method='average')

                        normalized_ranks = (size_ranks - 1) / (len(size_ranks) - 1)

                        label_to_color = {label: choose_color_models(label, norm_size) for label, norm_size in
                                          zip(x_labels, normalized_ranks)}

                        for type in ["zoom", "normal"]:
                            # Plot scatterplot
                            plt.figure(figsize=(6, 6))
                            for label, x_val, y_val in zip(x_labels, x_values, y_values):
                                plt.scatter(x_val, y_val, color=label_to_color[label], label=label, s=100)

                                # Add black horizontal line at horizontal_line_value
                                plt.axvline(x=get_baseline(task1_name), color='grey', linestyle='--', linewidth=0.5)

                                # Add black vertical line at vertical_line_value
                                plt.axhline(y=get_baseline(task2_name), color='grey', linestyle='--', linewidth=0.5)
                                plt.axis('square')
                            if type == "normal":
                                plt.xlim(0, 1)
                                plt.ylim(0, 1)
                            # Add legend
                            plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
                            if correlation_matrix is not None and p_values_matrix is not None:
                                corr_value = correlation_matrix.loc[task1_alias, task2_alias]
                                p_value = p_values_matrix.loc[task1_alias, task2_alias]
                                text = "Pearson's R: " if correlation_method == "pearson" else "Kendall's Tau: " if correlation_method == "kendall" else ""
                                text = f"{text}{corr_value:.2f}{'*' if p_value < 0.05 else ''}"
                                plt.gca().text(1.09, 0.50, text,
                                               transform=plt.gca().transAxes, fontsize="medium",
                                               bbox=dict(facecolor='white', edgecolor='#E0E0E0', boxstyle='square,pad=1'))

                            plt.xlabel(f"{task1_alias}")
                            plt.ylabel(f"{task2_alias}")
                            filename = f"{group1}_{task1_name}_vs_{group2}_{task2_name}.png".replace("/", "_")
                            output_path = output_path_root/type
                            output_path.mkdir(parents=True, exist_ok=True)

                            output_path = output_path/filename

                            plt.savefig(output_path, format="png", bbox_inches="tight")
                            plt.close()

                            pbar.update(1)

def build_and_save_scatterplots_models(scores: Dict, output_path_root: Path):
    total_iterations = len(scores.keys()) * len(scores.keys())
    with tqdm(total=total_iterations, desc="Building scatterplots", unit="scatterplot") as pbar:
        for model_name1, scores1 in scores.items():
            for model_name2, scores2 in scores.items():
                if model_name1 != model_name2:
                    x = scores1
                    y = scores2
                    x = [(s[0], s[1] / 100) if s[1] > 1 else (s[0], s[1]) for s in x]
                    y = [(s[0], s[1] / 100) if s[1] > 1 else (s[0], s[1]) for s in y]
                    x_labels, x_values = zip(*x)
                    y_labels, y_values = zip(*y)
                    label_to_color = {label: choose_color_benchmarks(label) for label in x_labels}

                    for type in ["zoom", "normal"]:
                        # Plot scatterplot
                        plt.figure(figsize=(10, 10))
                        for label, x_val, y_val in zip(x_labels, x_values, y_values):
                            #plt.scatter(x_val, y_val, color=label_to_color[label], label=label, s=100)
                            plt.plot([0, x_val], [0, y_val], linestyle='-', color=label_to_color[label], alpha=0.8, label=label)  # Draw line with endpoint
                            angle = np.degrees(np.arctan2(y_val, x_val))
                            # Position label slightly before the end
                            # rotation=angle,
                            plt.annotate(label,
                                         xy=(x_val, y_val),  # Point to connect
                                         xytext=(x_val + 0.09, y_val + 0.09),  # Offset position
                                         textcoords="data",
                                         fontsize=8,
                                         color="black",
                                         arrowprops=dict(arrowstyle="-", color="black", lw=1)
                                         # Black line without arrowhead
                                         )
                            #plt.text(x_val, y_val, label, fontsize=8,  va='center')  # Convert slope to degrees
                        if type == "normal":
                            plt.xlim(0, 1)
                            plt.ylim(0, 1)
                        # Add legend
                        plt.legend(title="Benchmarks", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

                        corr_value = 0.89
                        plt.figtext(1.05, 0.3, f"Correlation Value:\n{corr_value:.2f}", loc="upper left",
                                 fontsize="small")

                        plt.xlabel(f"{model_name1}")
                        plt.ylabel(f"{model_name2}")
                        filename = f"{model_name1}_vs_{model_name2}.png".replace("/", "_")
                        output_path = output_path_root / type
                        output_path.mkdir(parents=True, exist_ok=True)

                        output_path = output_path / filename

                        plt.savefig(output_path, format="png", bbox_inches="tight")
                        plt.close()

                        pbar.update(1)


def run_scatterplots(src_path: Path,
                    output_path_root:Path,
                     ignore_groups: List[str],
                     by: str,
                     correlation_method: str,
                     correlation_path: Path,
                     p_values_path: Path):
    src_path = project_root / src_path

    reports = get_reports(src_path=src_path)
    scores = get_scores(reports, benchmark_subset="main", take_above_baseline=False,
                        ignore_tasks=[], ignore_groups=ignore_groups, by=by)
    sort_scores(scores, by=by)

    correlation_matrix = pd.read_csv(correlation_path, index_col=0) if correlation_method != "" else None
    p_values_matrix = pd.read_csv(p_values_path, index_col=0) if correlation_method != "" else None
    if by=="benchmarks":
        build_and_save_scatterplots_benchmarks(scores=scores, output_path_root=output_path_root, correlation_matrix=correlation_matrix, correlation_method=correlation_method, p_values_matrix=p_values_matrix)
    elif by == "models":
        build_and_save_scatterplots_models(scores=scores, output_path_root=output_path_root)


