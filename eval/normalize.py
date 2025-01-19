from typing import Dict
from config import get_task_info

""" https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization """
def normalize_within_range(value, lower_bound, higher_bound):
    return (value - lower_bound) / (higher_bound - lower_bound)

def normalize_scores(results: Dict) -> None:
    num_tasks = len(results["task_results"])

    main_task_score = 0
    main_task_name = None
    for task_name, res in results["task_results"].items():
        try:
            _, task_config = get_task_info(task_name)
            score = res["score"]
            main_task = task_config["main_task"]
            random_baseline = task_config["random_baseline"]
            higher_bound = task_config["higher_bound"]

            if main_task:
                main_task_name = task_name

            if score < random_baseline:
                normalized_score = 0
            else:
                normalized_score = normalize_within_range(score, random_baseline, higher_bound)

            if num_tasks == 1:
                main_task_score = normalized_score
            else:
                if not main_task:
                    main_task_score += normalized_score
                    res["normalized_score"] = normalized_score
        except:
            print(f"Task {task_name} not included in the task registry. Ignoring..")
    if main_task_name == "fantom_full":
        # It's a special case because we don't aggregate according to subtasks in the traditional way. See FANToM Paper https://arxiv.org/pdf/2310.15421
        results["task_results"][main_task_name]["normalized_score"] = results["task_results"][main_task_name]["score"]
    else:
        results["task_results"][main_task_name]["normalized_score"] = main_task_score if num_tasks == 1 else main_task_score/(num_tasks - 1)

if __name__ == "__main__":
    import os
    import json

    # Define the folder containing JSON files
    folder_path = "/mnt/cimec-storage6/users/filippo.momente/PycharmProjects/playpen_eval/results/playpen/google__gemma-2-9b-it"

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .json extension
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)

            # Open and load the JSON file
            try:
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    normalize_scores(data)
                    print(f"Contents of {file_name}:")
                    print(data)
                    with open(file_path, 'w', encoding='utf-8') as json_file:
                        json.dump(data, json_file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_name}: {e}")
            except Exception as e:
                print(f"An error occurred with file {file_name}: {e}")


