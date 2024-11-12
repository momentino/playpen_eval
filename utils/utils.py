import torch
from pathlib import Path
from datetime import datetime

def custom_json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def get_all_json_paths(dir: Path):
    # Traverse all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            # Check if the file is a JSON file
            if filename.endswith('.json'):
                file_path = os.path.join(dirpath, filename)

                # Attempt to open and read the JSON file
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        print(f"Successfully read JSON from: {file_path}")
                        # Process the JSON data as needed
                        # For demonstration, print the loaded data
                        print(data)
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_path}: {e}")
                except Exception as e:
                    print(f"Could not open {file_path}: {e}")

def prepare_playpen_results(task_name: str, model_name:str, harness_results: dict = None) -> dict:
    results = {}
    if(harness_results is not None):
        subtask_results = {}

        # TODO Improve, support other scores
        task_score_key = [key for key in harness_results["results"][task_name] if ("acc" in key or "f1" in key) and "stderr" not in key]

        task_score_key = task_score_key[0]
        aggregated_metric_name = task_score_key.split(",")[0]
        aggregated_score_value = harness_results["results"][task_name][task_score_key]
        aggregated_results = {aggregated_metric_name: aggregated_score_value}

        results.update({
            "model_name": model_name,
            "task": task_name,
            "aggregated_results": aggregated_results,
            "subtask_results": subtask_results # TODO
        })
        return results
    raise Exception("Other options besides  harness are not yet implemented.")
