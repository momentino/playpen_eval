import json
import os
from typing import Dict, Any
from pathlib import Path
from collections import defaultdict
from frameworks.playeval_framework.models import Model
from frameworks.playeval_framework.tasks.task import Task

class WMTask(Task):
    def __init__(self):
        super().__init__(task_name="wm")
        self._prepare_dataset()
        self.experiments = ["grids_3*3", "grids_3*3_abstraction", "grids_3*3_abstraction_2", "grids_4*4", "grids_5*5", "grids_7*7", "letters"]

    def _prepare_dataset(self):
        dataset = defaultdict(lambda: defaultdict(list))
        data_path = Path(__file__).parent / 'wm_module' / "datasets"

        # get grids
        grids = defaultdict(dict)
        prompts = defaultdict(str)
        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                grandparent_folder = Path(file_path).parent.parent.name
                parent_folder = Path(file_path).parent.name
                if grandparent_folder == 'grids':
                    grid = json.load(open(file_path, 'r'))
                    grids[parent_folder].update(grid)
                if grandparent_folder == 'prompts':
                    with open(file_path, 'r') as f:
                        prompt = f.readline().strip()
                        prompts[parent_folder] = prompt

        try:
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)

                        parent_folder = Path(file_path).parent.name
                        grandparent_folder = Path(file_path).parent.parent.name

                        grid_task = False
                        if grandparent_folder != "grids" and grandparent_folder != "prompts":
                            if "grids" in grandparent_folder:
                                grid_task = True

                            with open(file_path, 'r') as f:
                                seq = f.readline().strip()
                                cond = f.readline().strip()

                            trials = []
                            for i in range(len(seq)):
                                trial = {}
                                if grid_task:
                                    trial['stimulus'] = grids[parent_folder][f"grid_{int(seq[i])}"]
                                trial['target'] = cond[i]
                                trial['response'] = ''
                                trial['correct'] = ''
                                trial['rt'] = ''
                                trials.append(trial)
                            dataset[grandparent_folder][parent_folder].append(trials)
            return dataset
        except Exception as e:
            print(f"An error occurred: {e}")

    def _get_prompt(self, exp_name: str) -> str:
        prompt_path = Path(__file__).parent / 'wm_module' / "datasets" / "prompts" / exp_name
        try:
            with open(prompt_path / 'prompt.txt', 'r') as f:
                prompt = f.readline().strip()
            return prompt
        except Exception as e:
            print(f"An error occurred: {e}")

    def evaluate(self, model: Model) -> Dict[str, Any]:
        for experiment in self.experiments:
            prompt = self._get_prompt(experiment)
