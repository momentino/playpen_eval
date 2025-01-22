""" Taken and adapted from the original repository (https://github.com/GU-CLASP/DNLI)"""
import csv
import guidance
from pathlib import Path
from typing import Dict, Any, List, NamedTuple, Tuple
from tqdm import tqdm
from frameworks.playpen_eval_benchmarks.models import Model
from frameworks.playpen_eval_benchmarks.tasks.task import Task
from guidance import select

ExampleDialogueUnlabelled = Tuple[List[Tuple[str, str]], str]
ExampleDialogueLabelled = Tuple[List[Tuple[str, str]], str, str]

class CompactDialogueSample(NamedTuple):
    index: int
    dialogue: List[Tuple[str, str]]
    hypothesis: str
    label: str

def split_dialogue(line: str) -> List[Tuple[str, str]]:
    turns = [t.split('>') for t in line.split('<')[1:]]
    return [(speaker, text.strip()) for (speaker, text) in turns]

def read_examples_dialogue(example_fn: str) -> List[ExampleDialogueLabelled]:
    with open(example_fn, "r", encoding="utf-8") as file:
        contents = [tuple(ln.strip().split('\t')) for ln in file.readlines()]
    return [(split_dialogue(dia), hyp, label) for (dia, hyp, label) in contents]

# The actual prompting for NLI
@guidance(stateless=True)
def nli_label(lm):
    return lm + select(["Entailment", "Contradiction", "Neutral"], name='label')

def prepare_prompts_nli_dialogue(prompt_instructions: str, examples: List[ExampleDialogueLabelled],
                                data: List[ExampleDialogueUnlabelled]) -> List[str]:
    formatted_examples = '\n'.join(["\n".join(["Speaker {}: {}".format(speaker, text) for (speaker, text) in turns]) +
                                     "\nHypothesis: {}\nRelation: {}".format(hyp, label)
                          for (turns, hyp, label) in examples])
    formatted_prompts = ["\n".join(["Speaker {}: {}".format(speaker, text) for (speaker, text) in turns]) +
                          "\nHypothesis: {}\nRelation: ".format(hyp) for (turns, hyp) in data]
    prompts = [prompt_instructions + '\n' + formatted_examples + '\n' + p for p in formatted_prompts]
    return prompts

class DNLI(object):
    def __init__(self, dnli_fn: str):
        self.dnli_fn = dnli_fn
        self.name = 'DNLI'
        self.data = self.load_data()

    def load_data(self) -> List[CompactDialogueSample]:
        with open(self.dnli_fn, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            items = [CompactDialogueSample(i, split_dialogue(row[0]), row[1], row[2]) for i, row in enumerate(reader)]
        return items

    def get_dialogue_hyp_pairs(self):
        return [(d.dialogue, d.hypothesis) for d in self.data]

    def get_labels(self):
        return [d.label for d in self.data]

class DNLITask(Task):
    def __init__(self):
        super().__init__(task_name="dnli")
        self._prepare_dataset()
        self._prepare_prompts()

    # TODO: allow to set more context lengths more easily
    def _prepare_dataset(self):
        data_path = Path(__file__).parent / "data" / "compiled" / "test_10_data.csv"
        self.dataset = DNLI(str(data_path))
        self.labels = self.dataset.get_labels()

    def _prepare_prompts(self) -> List[str]:
        self.instruction_prompt = "Given a dialogue excerpt and a Hypothesis, decide on the semantic relation between them, choosing between Entailment, Contradiction, and Neutral."
        self.items = self.dataset.get_dialogue_hyp_pairs()
        examples_path = Path(__file__).parent / "data" / "prompts" / "examples2.txt"
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



