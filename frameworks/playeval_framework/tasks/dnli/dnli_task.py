from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from enum import Enum
from frameworks.playeval_framework.tasks.dnli.dnli_module.baselines.llm.datasets import DNLI
from frameworks.playeval_framework.tasks.dnli.dnli_module.baselines.llm.main import prepare_prompts_nli_dialogue, \
    read_examples_dialogue, nli_label
from frameworks.playeval_framework.models import Model
from frameworks.playeval_framework.tasks.task import Task

class NLILabel(str, Enum):
    entailment = "Entailment"
    contradiction = "Contradiction"
    neutral = "Neutral"

""" Taken and adapted from the original repository (https://github.com/GU-CLASP/DNLI)"""
class DNLITask(Task):
    def __init__(self):
        super().__init__(task_name="dnli")
        self._prepare_dataset()
        self._prepare_prompts()

    # TODO: allow to set more context lengths more easily
    def _prepare_dataset(self):
        data_path = Path(__file__).parent / 'dnli_module' / "data" / "compiled" / "test_10_data.csv"
        self.dataset = DNLI(str(data_path))
        self.labels = self.dataset.get_labels()

    def _prepare_prompts(self) -> List[str]:
        self.instruction_prompt = "Given a dialogue excerpt and a Hypothesis, decide on the semantic relation between them, choosing between Entailment, Contradiction, and Neutral."
        self.items = self.dataset.get_dialogue_hyp_pairs()
        examples_path = Path(__file__).parent / "dnli_module" / "baselines" / "llm" / "prompts" / "examples2.txt"
        self.fewshot_examples = read_examples_dialogue(str(examples_path))
        self.prompts = prepare_prompts_nli_dialogue(self.instruction_prompt, self.fewshot_examples, self.items)


    def evaluate(self, model:Model) -> Dict[str, Any]:
        preds = []
        for prompt in tqdm(self.prompts):
            try:
                preds.append(model.generate_guidance([prompt, nli_label()])['label'])
            except AssertionError:
                print(prompt)

            # Count matching values at the same position
        acc = sum(1 for i in range(len(preds)) if preds[i] == self.labels[i])/len(self.labels)

        results = {"model_name": model.get_model_name().replace("/","__"),
                   "task_results": {}
                   }
        results["task_results"][self.task_name] = {"metric": "acc", "score": acc}
        return results



