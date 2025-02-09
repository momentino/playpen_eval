import json
import os
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from frameworks.playpen_eval_benchmarks.models import Model
from frameworks.playpen_eval_benchmarks.tasks.task import Task

class WMTask(Task):
    def __init__(self):
        super().__init__(task_name="wm")
        self.dataset = self._prepare_dataset()
        # "spatial_4*4", "spatial_5*5", "spatial_7*7", "spatial_3*3", "spatial_3*3_abstraction", "spatial_3*3_abstraction_2",
        self.experiments = ["verbal"]
        self.nback = 3

    def _prepare_dataset(self) -> Dict[str, Dict[str, Any]]:
        dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        data_path = Path(__file__).parent / "data"

        grids = defaultdict(dict)
        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                grandparent_folder = Path(file_path).parent.parent.name
                parent_folder = Path(file_path).parent.name
                if grandparent_folder == 'grids':
                    grid = json.load(open(file_path, 'r'))
                    grids[parent_folder].update(grid)

        try:
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        block = file.replace(".txt","")

                        type = Path(file_path).parent.parent.name
                        difficulty = Path(file_path).parent.name
                        grid_type = type if "abstraction" or "spatial_feedback" not in type else "grids_3*3"
                        grid_task = False
                        if type != "grids" and "prompts" not in file_path:
                            if "grids" in type:
                                grid_task = True
                                continue #TODO FIX
                            with open(file_path, 'r') as f:
                                seq = f.readline().strip().split(",")
                                cond = f.readline().strip().split(",")

                            trials = []
                            for i in range(len(seq)):
                                trial = {}
                                if grid_task:
                                    trial['stimulus'] = grids[grid_type][f"grid_{seq[i]}"]
                                else:
                                    trial['stimulus'] = seq[i]
                                trial['target'] = cond[i]
                                trials.append(trial)
                            dataset[type][difficulty][block] = trials
            return dataset
        except Exception as e:
            print(f"An error occurred: {e}")

    def _get_prompt(self, exp_name: str, n:int) -> str:
        nback = f"{str(n)}back"
        prompt_path = Path(__file__).parent / "data" / "prompts" / exp_name / nback
        try:
            with open(prompt_path / 'prompt.txt', 'r') as f:
                prompt = f.readline().strip()
            return prompt
        except Exception as e:
            print(f"An error occurred: {e}")

    def evaluate(self, model: Model, apply_chat_template:bool) -> Dict[str, Any]:
        out_of_context = 0
        exp_to_input_map = {
            "spatial_3*3": "grids_3*3",
            "spatial_3*3_abstraction": "grids_3*3_abstraction",
            "spatial_3*3_abstraction_2": "grids_3*3_abstraction_2",
            "spatial_4*4": "grids_4*4",
            "spatial_5*5": "grids_5*5",
            "spatial_7*7": "grids_7*7",
            "spatial_feedback": "grids_3*3",
            "verbal": "letters",
            "verbal_feedback": "letters"

        }
        results = defaultdict(lambda: defaultdict(float))
        for experiment in self.experiments:
            for n in range(1, self.nback+1):
                prompt = self._get_prompt(experiment, n)
                accuracies = []
                print("DATASET ",self.dataset)
                for block, trials in tqdm(self.dataset[exp_to_input_map[experiment]][f"{n}back"].items()):
                    correct = 0
                    messages = []
                    messages.append({"role": "system", "content": prompt})
                    for t in trials:
                        messages.append({"role": "user", "content": t["stimulus"]})
                        answer = model.generate(messages=messages, apply_chat_template=apply_chat_template)[0]
                        correct += answer == t["target"]
                    block_acc = correct / len(trials)
                    accuracies.append(block_acc)
                mean_acc = sum(accuracies) / len(accuracies)
                results[experiment][n] = mean_acc
        formatted_results = {"model_name": model.get_model_name().replace("/","__"),"out_of_context": out_of_context, "task_results": {}}

        total_accuracies = []
        for experiment, sub_exp in results.items():
            exp_accuracies = []
            for n, acc in sub_exp.items():
                formatted_results["task_results"][f"wm_{experiment}_{n}back"] = {"metric": "acc", "score": acc}
                exp_accuracies.append(acc)
                total_accuracies.append(acc)
            formatted_results["task_results"][f"wm_{experiment}_overall"] = {"metric": "acc", "score": sum(exp_accuracies) / len(exp_accuracies)}
        formatted_results["task_results"]["wm"] = {"metric": "acc",
                                                                      "score": sum(total_accuracies) / len(
                                                                          total_accuracies)}
        return formatted_results
