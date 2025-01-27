import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.cm import get_cmap
from pathlib import Path
from typing import Dict, List
from config import get_model_registry, get_task_registry, project_root, get_alias, get_baseline
from analyze.score_extraction_utils import get_reports, get_scores, sort_scores


def build_and_save_scatterplots(scores: Dict, output_path_root: Path):
    processed_pairs = set()

    total_iterations = sum(len(tasks1) * len(tasks2) for tasks1 in scores.values() for tasks2 in scores.values())
    with tqdm(total=total_iterations, desc="Building scatterplots", unit="scatterplot") as pbar:
        for group1, tasks1 in scores.items():
            for task1_name, scores1 in tasks1.items():
                for group2, tasks2 in scores.items():
                    for task2_name, scores2 in tasks2.items():
                        pair = tuple(sorted([task1_name, task2_name]))
                        if task1_name != task2_name and pair not in processed_pairs:
                            processed_pairs.add(pair)
                            x = scores1
                            y = scores2
                            x_labels, x_values = zip(*x)
                            y_labels, y_values = zip(*y)
                            # Ensure labels match between x and y
                            assert x_labels == y_labels, f"Labels in x {x_labels} and y {y_labels} arrays do not match!"

                            # Assign unique colors to each label
                            unique_labels = list(set(x_labels))  # Find unique labels
                            color_map = get_cmap("tab10", len(unique_labels))  # Use a colormap
                            label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

                            for type in ["zoom", "normal"]:
                                # Plot scatterplot
                                plt.figure(figsize=(10, 6))
                                for label, x_val, y_val in zip(x_labels, x_values, y_values):
                                    plt.scatter(x_val, y_val, color=label_to_color[label], label=label, s=100)

                                    # Add black horizontal line at horizontal_line_value
                                    plt.axvline(x=get_baseline(task1_name), color='black', linestyle='-', linewidth=2)

                                    # Add black vertical line at vertical_line_value
                                    plt.axhline(y=get_baseline(task2_name), color='black', linestyle='-', linewidth=2)
                                if type == "normal":
                                    plt.xlim(0, 1)
                                    plt.ylim(0, 1)
                                # Add legend
                                plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")



                                plt.xlabel(f"{get_alias(task1_name)}")
                                plt.ylabel(f"{get_alias(task2_name)}")
                                filename = f"{group1}_{task1_name}_vs_{group2}_{task2_name}.png".replace("/", "_")
                                output_path = output_path_root/type
                                output_path.mkdir(parents=True, exist_ok=True)

                                output_path = output_path/filename

                                plt.savefig(output_path, format="png", bbox_inches="tight")
                                plt.close()

                                pbar.update(1)



def run_scatterplots(src_path: Path,
                    output_path_root:Path,
                     ignore_groups: List[str]):
    model_registry = get_model_registry()
    task_registry = get_task_registry()
    src_path = project_root / src_path

    reports = get_reports(src_path=src_path, model_registry=model_registry)
    scores = get_scores(reports, task_registry, subset="all", take_above_baseline=False,
                        ignore_tasks=[], ignore_groups=ignore_groups)
    sort_scores(scores)
    build_and_save_scatterplots(scores=scores, output_path_root=output_path_root)


