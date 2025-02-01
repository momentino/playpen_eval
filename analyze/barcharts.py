import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from config import project_root, get_task_info, MODEL_REGISTRY, TASK_REGISTRY
from analyze.score_extraction_utils import get_reports, get_scores, sort_scores

def set_label_color(task_name: str):
    _, task_info = get_task_info(task_name)
    category = task_info["functional_group"][0]
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
            values_to_plot = [item[1] for item in scores]
            custom_labels = [f'{scores[i][0]}' for i in range(len(values_to_plot))]  # Custom label for each bin
            plt.figure(figsize=(8,6))
            colors = [set_label_color(label) for label in custom_labels]
            plt.bar(custom_labels, values_to_plot, alpha=0.7, edgecolor='black', color=colors)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'{model_name}')
            plt.tight_layout()
            output_path = output_path_root / model_name
            plt.savefig(f'{output_path}.png')
            plt.close()
            pbar.update(1)

def run_barcharts(src_path: Path,
                    output_path_root:Path,
                     ignore_groups: List[str],
                     by: str):
    model_registry = get_model_registry()
    task_registry = get_task_registry()
    src_path = project_root / src_path
    output_path_root.mkdir(parents=True, exist_ok=True)
    reports = get_reports(src_path=src_path, model_registry=model_registry)
    scores = get_scores(reports, task_registry, benchmark_subset="main", take_above_baseline=False,
                        ignore_tasks=[], ignore_groups=ignore_groups, by=by)
    sort_scores(scores, by=by)
    if by == "benchmarks":
        build_and_save_barcharts(scores=scores, output_path_root=output_path_root)
    elif by == "models":
        build_and_save_barcharts(scores=scores, output_path_root=output_path_root)