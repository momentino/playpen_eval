""" Taken from the original work: https://github.com/kenneds6/LLM-cognitive-flexibility/blob/main/src/tests/lnt.py"""

"""
Letter Number Test implementation.
"""
import json
import random
import string
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Literal

Task = Literal['letter', 'number']
Response = Literal['vowel', 'consonant', 'odd', 'even']


@dataclass
class LNTConfig:
    num_trials: int = 25
    num_successes_before_switch: int = 6


class LNT:
    def __init__(self, config: LNTConfig = LNTConfig()):
        self.config = config
        self.current_task = random.choice(['letter', 'number'])
        self.score = 0
        self.successes = 0
        self.trials = 0

    def generate_sequence(self) -> str:
        """Generate a random letter-number sequence."""
        letter = random.choice(string.ascii_letters)
        number = random.randint(0, 9)
        return f"{letter}{number}"

    def _is_vowel(self, letter: str) -> bool:
        """Check if a letter is a vowel."""
        return letter.lower() in 'aeiou'

    def _is_even(self, number: int) -> bool:
        """Check if a number is even."""
        return number % 2 == 0

    def evaluate_response(self, sequence: str, response: Response) -> bool:
        """Evaluate if the response is correct for the current task."""
        letter, number = sequence[0], int(sequence[1])

        if self.current_task == 'letter':
            is_correct = ((self._is_vowel(letter) and response == 'vowel') or
                          (not self._is_vowel(letter) and response == 'consonant'))
        else:  # number task
            is_correct = ((self._is_even(number) and response == 'even') or
                          (not self._is_even(number) and response == 'odd'))

        if is_correct:
            self.score += 1
            self.successes += 1
        else:
            self.successes = 0

        # Check if task should switch
        if self.successes == self.config.num_successes_before_switch:
            self.current_task = 'letter' if self.current_task == 'number' else 'number'
            self.successes = 0

        self.trials += 1
        return is_correct

    def get_performance(self) -> Tuple[float, int, int]:
        """Return accuracy, number of successes, and total trials."""
        accuracy = self.score / self.trials if self.trials > 0 else 0
        return accuracy, self.score, self.trials

class LNTRevisited(LNT):
    def __init__(self, eval_num: int, config: LNTConfig = LNTConfig()):
        super().__init__(config)
        self.config = config
        self.score = 0
        self.successes = 0
        self.trials = 0

        rules_successors_path = Path(__file__).parent / "revisited_data" / 'lnt' / 'rules.json'

        self.rules_successors = json.load(open(rules_successors_path, 'r'))
        self.current_task = self.rules_successors["starter"][eval_num]