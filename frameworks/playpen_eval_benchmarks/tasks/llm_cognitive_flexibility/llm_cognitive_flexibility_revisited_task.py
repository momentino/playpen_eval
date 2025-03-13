import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from frameworks.playpen_eval_benchmarks.models import Model, HF
from frameworks.playpen_eval_benchmarks.tasks.task import Task
from frameworks.playpen_eval_benchmarks.tasks.llm_cognitive_flexibility.wcst_test import WCSTConfig, WCSTRevisited
from frameworks.playpen_eval_benchmarks.tasks.llm_cognitive_flexibility.lnt_test import LNTConfig, LNTRevisited

class LLMCognitiveFlexibilityRevisitedTask(Task):
    def __init__(self):
        super().__init__(task_name="llm_cognitive_flexibility_revisited")

    def _format_card(self, card: tuple) -> str:
        """Format card tuple as string."""
        return f"{card[0]} {card[1]} {card[2]}"

    def _extract_choice(self, response: str) -> Optional[int]:
        """Extract numerical choice from response."""
        if "option" in response.lower():
            match = re.search(r'option\s?(\d+)', response, re.IGNORECASE)
            if match:
                return int(match.group(1)) - 1
        try:
            return int(response.strip()) - 1
        except ValueError:
            return None

    def _extract_ln_response(self, response: str) -> Optional[str]:
        """Extract letter-number task response."""
        matches = re.findall(r"vowel|consonant|even|odd", response.lower())
        return matches[0] if matches else None

    def evaluate_wcst(self, model: Model | HF, apply_chat_template:bool, data: List[List[Dict]]):
        WCST_SYSTEM_PROMPT = """
        You are participating in a card matching exercise.
        For each trial, you will be presented with a card and four option cards.
        Your task is to match the presented card with one of the options by responding with just the number (1-4).
        There is always a correct way to match the cards, but you will need to discover it through trial and error.
        When your match is correct, continue using the same matching approach until you receive feedback that it's incorrect.
        When incorrect, you must switch to a completely different matching approach - do not persist with an approach that failed.
        Respond only with a single number between 1 and 4.
        Do not explain your choice or thought process.
        """
        config: WCSTConfig = WCSTConfig()
        results = []
        for eval_num, evaluation_instances in enumerate(data):
            test = WCSTRevisited(eval_num=eval_num, config=config)
            messages = [{"role": "system", "content": WCST_SYSTEM_PROMPT}]
            for instance in evaluation_instances:
                card = instance['card']
                options = instance['options']
                print(" CARD ", card, "OPTIONS ",options)
                prompt = f"\nNew Card: {self._format_card(card)}\n"
                for i, option in enumerate(options, 1):
                    prompt += f"Option {i}: {self._format_card(option)}\n"
                prompt += "Choose the correct option (1-4): "

                messages.append({"role":"user","content":prompt})
                response = model.generate(messages, apply_chat_template=apply_chat_template)
                choice = self._extract_choice(response[0])

                if choice is None:
                    print(f"Invalid response format: {response}")
                    continue

                is_correct = test.evaluate_choice(card, choice, options)
                feedback = "Correct!" if is_correct else "Incorrect!"

                messages.append({"role":"user","content":feedback})

            accuracy, num_successes, trials = test.get_performance()
            eval_result = {
                "evaluation": eval_num + 1,
                "accuracy": accuracy,
                "num_successes": num_successes,
                "trials": trials
            }
            results.append(eval_result)
        return results

    def evaluate_lnt(self, model: Model | HF, apply_chat_template:bool, data: List[List[Dict]]):
        LNT_SYSTEM_PROMPT = """
        You are participating in a sequence classification exercise.
        For each trial, you will see a sequence containing one letter followed by one number.
        Your task is to classify the sequence in one of two ways:
        For letters: respond with 'vowel' or 'consonant'
        For numbers: respond with 'even' or 'odd'
        You must choose ONE type of classification and stick with it while it works.
        If you receive incorrect feedback, you must switch to the other classification task - do not persist with a failed approach.
        Respond only with a single word: 'vowel', 'consonant', 'even', or 'odd'.
        Do not explain your choice or provide both classifications.
        """
        config: LNTConfig = LNTConfig()
        results = []
        for eval_num, evaluation_instances in enumerate(data):
            print(" EVAL NUM ",eval_num)
            test = LNTRevisited(eval_num=eval_num, config=config)
            messages = [{"role": "system", "content": LNT_SYSTEM_PROMPT}]
            for instance in evaluation_instances:
                print(" SEQUENCE ", sequence)
                sequence = instance['sequence']
                prompt = f"\nSequence: {sequence}\n"

                messages.append({"role": "user", "content": prompt})
                response = model.generate(messages, apply_chat_template=apply_chat_template)
                choice = self._extract_ln_response(response[0])

                if choice is None:
                    print(f"Invalid response format: {response}")
                    continue

                is_correct = test.evaluate_response(sequence, choice)
                feedback = "Correct!" if is_correct else "Incorrect!"

                messages.append({"role":"user","content":feedback})
                accuracy, num_successes, trials = test.get_performance()
                eval_result = {
                    "evaluation": eval_num + 1,
                    "accuracy": accuracy,
                    "num_successes": num_successes,
                    "trials": trials
                }
                results.append(eval_result)
            return results

    def evaluate(self, model: Model | HF, apply_chat_template:bool) -> Dict[str, Any]:
        num_evaluations = 8
        num_trials = 25
        wcst_data_path = Path(__file__).parent / "revisited_data" / 'wcst' / 'data.json'
        lnt_data_path = Path(__file__).parent / "revisited_data" / 'lnt' / 'data.json'
        wcst_data = json.load(open(wcst_data_path,'r'))
        lnt_data = json.load(open(lnt_data_path,'r'))
        wcst_results = self.evaluate_wcst(model,apply_chat_template, wcst_data)
        wcst_acc = sum([r["accuracy"] for r in wcst_results])/len(wcst_results)
        lnt_results = self.evaluate_lnt(model, apply_chat_template, lnt_data)
        lnt_acc = sum([r["accuracy"] for r in lnt_results]) / len(lnt_results)


        formatted_results = {"model_name": model.get_model_name().replace("/", "__"), "task_results": {}}
        formatted_results["task_results"]["llm_cognitive_flexibility_wcst_revisited"] = {"metric": "acc",
                                                                               "score": wcst_acc}
        formatted_results["task_results"]["llm_cognitive_flexibility_lnt_revisited"] = {"metric": "acc",
                                                                               "score": lnt_acc}
        formatted_results["task_results"]["llm_cognitive_flexibility_revisited"] = {"metric": "acc",
                                                                              "score": (wcst_acc+lnt_acc)/2}

        return formatted_results

