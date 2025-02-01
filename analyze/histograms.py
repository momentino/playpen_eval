import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from config import get_model_registry, get_task_registry, project_root, get_alias, get_baseline, get_task_info
from analyze.score_extraction_utils import get_reports, get_scores, sort_scores

def build_and_save_histograms(scores: Dict, output_path_root: Path):
    total_iterations = len(scores.keys()) * len(scores.keys())
    with tqdm(total=total_iterations, desc="Building histograms", unit="histogram") as pbar:
        for model_name, scores in scores.items():
            values_to_plot = [item[1] for item in scores]
            print(" VALUES TO PLOT ",values_to_plot)
            # Create a histogram for each key
            plt.figure()  # Create a new figure for each key
            print(" N BINS ",len(scores))
            values_to_plot = [1,2,3,4,5]
            n, bins, patches = plt.hist(values_to_plot, bins=len(values_to_plot), alpha=0.7, edgecolor='black')  # You can adjust bins as needed
            print(patches)
            plt.title(f'Histogram for {model_name}')
            plt.ylabel('Performance')
            bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate the center of each bin
            custom_labels = [f'{scores[i][0]}' for i in range(len(bin_centers))]  # Custom label for each bin
            plt.xticks(bin_centers, labels=custom_labels, rotation=45)  # Rotate labels for better readability
            plt.grid(True)
            output_path = output_path_root / model_name
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{output_path}.png')  # Save as a PNG file (you can change the file type)
            plt.close()  # Close the figure to avoid display repetition

def run_histograms(src_path: Path,
                    output_path_root:Path,
                     ignore_groups: List[str],
                     by: str):
    model_registry = get_model_registry()
    task_registry = get_task_registry()
    src_path = project_root / src_path

    reports = get_reports(src_path=src_path, model_registry=model_registry)
    scores = get_scores(reports, task_registry, benchmark_subset="main", take_above_baseline=False,
                        ignore_tasks=[], ignore_groups=ignore_groups, by=by)
    sort_scores(scores, by=by)
    if by == "benchmarks":
        build_and_save_histograms(scores=scores, output_path_root=output_path_root)
    elif by == "models":
        build_and_save_histograms(scores=scores, output_path_root=output_path_root)