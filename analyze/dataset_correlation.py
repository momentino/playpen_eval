from analyze import project_root

def run_correlation(results_path: str):
    results_grouped_by_task = {}

    results_path = project_root / results_path