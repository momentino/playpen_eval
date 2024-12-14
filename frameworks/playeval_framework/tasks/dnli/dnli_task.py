from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
from frameworks.playeval_framework.tasks.dnli.dnli_module.baselines.llm.datasets import DNLI
from frameworks.playeval_framework.tasks.dnli.dnli_module.baselines.llm.main import prepare_prompts_nli_dialogue, prompt_model
from frameworks.playeval_framework.models import Model
from frameworks.playeval_framework.tasks.task import Task

""" Taken and adapted from the original repository (add source)"""
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

    def _prepare_prompts(self):
        self.instruction_prompt = "Given a dialogue excerpt and a Hypothesis, decide on the semantic relation between them, choosing between Entailment, Contradiction, and Neutral."
        self.items = self.dataset.get_dialogue_hyp_pairs()
        examples_path = Path(__file__).parent / "dnli_module" / "baselines" / "llm" / "prompts" / "examples2.csv"
        self.fewshot_examples = DNLI(str(examples_path)).get_dialogue_hyp_pairs()
        self.prompts = prepare_prompts_nli_dialogue(self.instruction_prompt, self.fewshot_examples, self.items)

    def evaluate(self, model:Model) -> Dict[str, Any]:
        preds = []
        for prompt in tqdm(self.prompts):
            try:
                preds.append(prompt_model(model, prompt))
            except AssertionError:
                print(prompt)
        print(" PREDS ", preds[:10])
        print(" LABELS ", self.labels[:10])

            # Count matching values at the same position
        acc = sum(1 for i in range(len(preds)) if preds[i] == self.labels[i])/len(self.labels)

        results = {"model_name": model.get_model_name(),
                   "task:": self.task_name,
                   "aggregated_results": {"metric":"acc", "score":acc},
                   "subtasks":{}
                   }
        return results



