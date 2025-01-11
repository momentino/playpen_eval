from typing import Dict, Any, List
from pathlib import Path

from tqdm import tqdm

from frameworks.playeval_framework.models import Model
from frameworks.playeval_framework.tasks.task import Task
from game import Q20Game, Q20GameCelebrity

DEFAULT_TEMPERATURE = 0.8  # That's the default from the original dataset


class EDATask(Task):
    def __init__(self):
        super().__init__(task_name="eda")
        self._prepare_things_dataset()
        self._prepare_celebrities_dataset()

    def _prepare_things_dataset(self):
        with open(Path(__file__).parent / 'data' / "newlist_things.rmdup.train.txt",
                  "r", encoding="utf-8") as f_input:
            self.things_dataset = [l.strip() for l in f_input.readlines()]

    def _prepare_celebrities_dataset(self):
        with open(Path(__file__).parent / 'data' / "newlist_celebs.rmdup.train.txt",
                  "r", encoding="utf-8") as f_input:
            self.celebrities_dataset = [l.strip() for l in f_input.readlines()]

    def evaluate(self, model: Model) -> Dict[str, Any]:
        results = {
            "model_name": model.get_model_name().replace("/", "__"),
            "task": self.task_name
        }
        agg = 0

        guesser_kargs = {
            "max_new_tokens": 64,
            "temperature": DEFAULT_TEMPERATURE,
            "repetition_penalty": 1.0,
            "do_sample": True,
        }

        successes = 0
        for item in tqdm(self.things_dataset):
            game = Q20Game(
                item=item,
                answerer_model=model,
                guesser_model=model,
                guesser_tokenizer=model.get_tokenizer(),
                num_turns=20,
                temperature=DEFAULT_TEMPERATURE,
                guesser_kargs=guesser_kargs,
            )
            if game.game_play(False):
                successes += 1

        results["subtask_results"]["things"] = {
            "metric": "acc",
            "score": successes / len(self.things_dataset)
        }
        agg += successes / len(self.things_dataset)

        successes = 0
        for item in tqdm(self.celebrities_dataset):
            game = Q20GameCelebrity(
                item=item,
                answerer_model=model,
                guesser_model=model,
                guesser_tokenizer=model.get_tokenizer(),
                num_turns=20,
                temperature=DEFAULT_TEMPERATURE,
                guesser_kargs=guesser_kargs,
            )
            if game.game_play(False):
                successes += 1

        results["subtask_results"]["celebrities"] = {
            "metric": "acc",
            "score": successes / len(self.celebrities_dataset)
        }
        agg += successes / len(self.things_dataset)

        results["aggregated_results"] = {
            "metric": "acc",
            "score": agg / 2
        }

        return results


