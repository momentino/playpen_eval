from typing import Dict, Any, List
from pathlib import Path

from tqdm import tqdm

from frameworks.playpen_eval_benchmarks.models import Model, HF
from frameworks.playpen_eval_benchmarks.tasks.task import Task
from frameworks.playpen_eval_benchmarks.tasks.entity_deduction_arena.game import Q20Game, Q20GameCelebrity

DEFAULT_TEMPERATURE = 0.8  # That's the default from the original dataset


class EDATask(Task):
    def __init__(self):
        super().__init__(task_name="entity_deduction_arena")
        self._prepare_things_dataset()
        self._prepare_celebrities_dataset()

    def _prepare_things_dataset(self):
        with open(Path(__file__).parent / 'data' / "things" / "newlist_things.rmdup.train.txt",
                  "r", encoding="utf-8") as f_input:
            self.things_dataset = [l.strip() for l in f_input.readlines()]

    def _prepare_celebrities_dataset(self):
        with open(Path(__file__).parent / 'data' / "celebrities" / "newlist_celebs.all.rmdup.train.txt",
                  "r", encoding="utf-8") as f_input:
            self.celebrities_dataset = [l.strip() for l in f_input.readlines()]

    def evaluate(self, model: Model, apply_chat_template: bool) -> Dict[str, Any]:
        results = {
            "model_name": model.get_model_name().replace("/", "__")
        }
        agg = 0

        answerer_model = HF(pretrained = 'meta-llama/Meta-Llama-3.1-70B-Instruct',
                 device = model.get_device(),
                 trust_remote_code = True,
                 torch_dtype='bfloat16',
                 gen_kwargs= {'temperature':0.2}
                 )

        successes = 0
        for item in tqdm(self.things_dataset):
            game = Q20Game(
                item=item,
                answerer_model=answerer_model,
                guesser_model=model,
                num_turns=20,
                apply_chat_template=apply_chat_template
            )
            if game.game_play(False):
                successes += 1

        results["task_results"] = {}
        results["task_results"]["entity_deduction_arena_things"] = {
            "metric": "acc",
            "score": successes / len(self.things_dataset)
        }
        agg += successes / len(self.things_dataset)

        successes = 0
        for item in tqdm(self.celebrities_dataset):
            game = Q20GameCelebrity(
                item=item,
                answerer_model=answerer_model,
                guesser_model=model,
                num_turns=20,
                apply_chat_template=apply_chat_template
            )
            if game.game_play(False):
                successes += 1

        results["task_results"]["entity_deduction_arena_celebrities"] = {
            "metric": "acc",
            "score": successes / len(self.celebrities_dataset)
        }
        agg += successes / len(self.things_dataset)

        results["task_results"]["entity_deduction_arena"] = {
            "metric": "acc",
            "score": agg / 2
        }

        return results


