import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from config import project_root, get_task_info, get_capability_group_from_task_name, get_alias, get_capability_alias, \
    get_capability_group_from_alias
from analyze.score_extraction_utils import get_reports, get_scores, sort_scores

def set_label_color(task_alias: str):
    category = get_capability_group_from_alias(task_alias)
    if category == "core_executive_functions":
        color="darkred"
    if category == "executive_functions":
        color="Red"
    elif category == "social_emotional_cognition":
        color="Blue"
    elif category == "massive":
        color="Green"
    elif category == "interactive":
        color="Purple"
    return color

def build_and_save_barcharts(scores: Dict, output_path_root: Path):
    total_iterations = len(scores.keys())
    with tqdm(total=total_iterations, desc="Building bar charts", unit="bar charts") as pbar:

        for model_name, scores in scores.items():
            print(model_name)
            print(scores)
            values_to_plot = [item[1] for item in scores]
            ceilings = [get_task_info(item[0])[1]["random_baseline"] for item in scores]
            custom_labels = [get_alias(item[0]) for item in scores]
            #custom_labels = [get_alias(item[0]) if get_task_info(item[0])[1]['category'] in ['massive, interactive'] else get_capability_alias(get_task_info(item[0])[1]['category']) for item in scores]
            fig, ax = plt.subplots(figsize=(8,6))
            colors = [set_label_color(label) for label in custom_labels]

            x = np.arange(len(custom_labels))

            # Width of the bars
            width = 0.35

            ax.bar(x -width/2, custom_labels, values_to_plot, alpha=0.7, edgecolor='black', color=colors)
            ax.bar(x + width/2, custom_labels, values_to_plot, alpha=0.7, edgecolor='green', color=colors)
            for i, ceiling in enumerate(ceilings):
                ax.hlines(y=ceiling, xmin=i - 0.4, xmax=i + 0.4, color='black', linestyle='--', linewidth=2)
            plt.xticks(rotation=45, ha='right')
            ax.set_title(f'{model_name}')
            fig.tight_layout()
            output_path = output_path_root / model_name
            plt.savefig(f'{output_path}.png')
            plt.close()
            pbar.update(1)

def run_barcharts(src_path: Path,
                    output_path_root:Path,
                     ignore_groups: List[str],
                     by: str):
    src_path = project_root / src_path
    output_path_root.mkdir(parents=True, exist_ok=True)
    reports = get_reports(src_path=src_path)
    scores = get_scores(reports, benchmark_subset="main", take_above_baseline=False,
                        ignore_tasks=[], ignore_groups=ignore_groups, by=by)
    sort_scores(scores, by=by)
    if by == "benchmarks":
        build_and_save_barcharts(scores=scores, output_path_root=output_path_root)
    elif by == "models":
        build_and_save_barcharts(scores=scores, output_path_root=output_path_root)