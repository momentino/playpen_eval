""" Taken from the original work: https://github.com/kenneds6/LLM-cognitive-flexibility/blob/main/src/tests/wcst.py """

"""
Wisconsin Card Sorting Test implementation.
"""
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Literal

Rule = Literal['shape', 'color', 'number']
Card = Tuple[str, str, int]

@dataclass
class WCSTConfig:
    num_trials: int = 25
    num_successes_before_switch: int = 6
    shapes: List[str] = field(default_factory=lambda: ["circle", "triangle", "cross", "star"])
    colors: List[str] = field(default_factory=lambda: ["red", "green", "blue", "yellow"])
    numbers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

class WCST:
    def __init__(self, config: WCSTConfig = WCSTConfig()):
        self.config = config
        self.deck = self._generate_deck()
        random.shuffle(self.deck)
        self.current_rule = random.choice(['shape', 'color', 'number'])
        self.score = 0
        self.successes = 0

        self.rule_iterators = {
            'shape': 0,
            'color': 0,
            'number': 0
        }
        rules_successors_path = Path(__file__).parent / "revisited_data" / 'wcst' / 'rules.json'

        self.rules_successors = json.load(open(rules_successors_path, 'r'))

    def _generate_deck(self) -> List[Card]:
        """Generate all possible card combinations."""
        deck = [(shape, color, number)
               for shape in self.config.shapes
               for color in self.config.colors
               for number in self.config.numbers]
        return deck * 5  # Increase the deck size

    def generate_options(self, card: Card) -> List[Card]:
        """Generate four option cards based on the input card."""
        options = []
        shapes = self.config.shapes.copy()
        colors = self.config.colors.copy()
        numbers = self.config.numbers.copy()

        card_shape = card[0]
        card_color = card[1]
        card_number = card[2]

        # card 1
        colors_copy = colors.copy()
        colors_copy.remove(card_color)
        card_color_1 = random.choice(colors_copy)

        numbers_copy = numbers.copy()
        numbers_copy.remove(card_number)
        card_number_1 = random.choice(numbers_copy)

        card_1 = (card_shape, card_color_1, card_number_1)
        options.append(card_1)
        shapes.remove(card_shape)
        colors.remove(card_color_1)
        numbers.remove(card_number_1)

        # card 2
        shapes_copy = shapes.copy()
        if card_shape in shapes_copy:
            shapes_copy.remove(card_shape)
        card_shape_2 = random.choice(shapes_copy)

        numbers_copy = numbers.copy()
        numbers_copy.remove(card_number)
        card_number_2 = random.choice(numbers_copy)

        card_2 = (card_shape_2, card_color, card_number_2)
        options.append(card_2)
        shapes.remove(card_shape_2)
        colors.remove(card_color)
        numbers.remove(card_number_2)

        # card 3
        shapes_copy = shapes.copy()
        if card_shape in shapes_copy:
            shapes_copy.remove(card_shape)
        card_shape_3 = random.choice(shapes_copy)

        colors_copy = colors.copy()
        if card_color in colors_copy:
            colors_copy.remove(card_color)
        card_color_3 = random.choice(colors_copy)

        card_3 = (card_shape_3, card_color_3, card_number)
        options.append(card_3)
        shapes.remove(card_shape_3)
        colors.remove(card_color_3)
        numbers.remove(card_number)

        # card 4
        shapes_copy = shapes.copy()
        if card_shape in shapes_copy:
            shapes_copy.remove(card_shape)
        card_shape_4 = random.choice(shapes_copy)

        colors_copy = colors.copy()
        if card_color in colors_copy:
            colors_copy.remove(card_color)
        card_color_4 = random.choice(colors_copy)

        numbers_copy = numbers.copy()
        if card_number in numbers_copy:
            numbers_copy.remove(card_number)
        card_number_4 = random.choice(numbers_copy)

        card_4 = (card_shape_4, card_color_4, card_number_4)
        options.append(card_4)

        random.shuffle(options)
        return options

    def evaluate_choice(self, card: Card, choice: int, options: List[Card]) -> bool:
        """Evaluate the choice based on the current rule."""
        rule_indices = {"shape": 0, "color": 1, "number": 2}

        # Check if the chosen card matches the current rule
        chosen_rule = self.current_rule if options[choice][rule_indices[self.current_rule]] == card[rule_indices[self.current_rule]] else None

        if chosen_rule == self.current_rule:
            self.score += 1
            self.successes += 1
        else:
            self.successes = 0

        # Change the rule after 6 successes
        if self.successes == 6:
            rule_index = self.rule_iterators[self.current_rule]
            self.current_rule = self.rules_successors[self.current_rule][rule_index]
            self.successes = 0

        return chosen_rule == self.current_rule

    def evaluate_choice(self, card: Card, choice: int, options: List[Card]) -> bool:
        """Evaluate the choice based on the current rule."""
        rule_indices = {"shape": 0, "color": 1, "number": 2}

        # Check if the chosen card matches the current rule
        chosen_rule = self.current_rule if options[choice][rule_indices[self.current_rule]] == card[
            rule_indices[self.current_rule]] else None

        if chosen_rule == self.current_rule:
            self.score += 1
            self.successes += 1
        else:
            self.successes = 0

        # Change the rule after 6 successes
        if self.successes == 6:
            rules = ["shape", "color", "number"]
            rules.remove(self.current_rule)
            self.current_rule = random.choice(rules)
            self.successes = 0

        return chosen_rule == self.current_rule

    def run_task(self, verbose: bool = False):
        """Run the complete WCST task."""
        deck = self.deck.copy()
        for i, card in enumerate(deck):
            # Remove the current card from the deck
            deck.pop(deck.index(card))

            # Regenerate lists for each iteration
            shapes = self.config.shapes.copy()
            colors = self.config.colors.copy()
            numbers = self.config.numbers.copy()

            # Generate options
            options = self.generate_options(card)

            # Print current rule if verbose
            if verbose:
                print(f"Current rule: {self.current_rule}")

            # Simulate correct choice (as in the original script)
            rule_indices = {"shape": 0, "color": 1, "number": 2}
            for j, option in enumerate(options):
                if option[rule_indices[self.current_rule]] == card[rule_indices[self.current_rule]]:
                    choice = j
                    break

            # Evaluate the choice
            is_correct = self.evaluate_choice(card, choice, options)

            if verbose:
                print(f"Card: {card}")
                print("Options:")
                for k, opt in enumerate(options, 1):
                    print(f"Option {k}: {opt}")
                print(f"Chosen option: {options[choice]}")
                print("Correct!" if is_correct else "Incorrect!")
                print(f"Score: {self.score}")
                print(f"Successes: {self.successes}")
                print("---")

            # Stop if we've reached the specified number of trials
            if i >= self.config.num_trials - 1:
                break

        return self.score

    def get_performance(self):
        """Return accuracy, number of successes, and total trials."""
        return (self.score / self.config.num_trials, self.score, self.config.num_trials)

class WCSTRevisited(WCST):
    def __init__(self, eval_num: int, config: WCSTConfig = WCSTConfig()):
        super().__init__(config)
        self.config = config
        self.deck = self._generate_deck()
        #random.shuffle(self.deck)
        #self.current_rule = random.choice(['shape', 'color', 'number'])
        self.score = 0
        self.successes = 0

        self.eval_num = eval_num
        self.rule_iterators = {
            'shape': 0,
            'color': 0,
            'number': 0
        }
        rules_successors_path = Path(__file__).parent / "revisited_data" / 'wcst' / 'rules.json'

        self.rules_successors = json.load(open(rules_successors_path, 'r'))
        self.current_rule = self.rules_successors["starter"][eval_num]

    def evaluate_choice(self, card: Card, choice: int, options: List[Card]) -> bool:
        """Evaluate the choice based on the current rule."""
        rule_indices = {"shape": 0, "color": 1, "number": 2}

        # Check if the chosen card matches the current rule
        chosen_rule = self.current_rule if options[choice][rule_indices[self.current_rule]] == card[rule_indices[self.current_rule]] else None

        if chosen_rule == self.current_rule:
            self.score += 1
            self.successes += 1
        else:
            self.successes = 0

        # Change the rule after 6 successes
        if self.successes == 6:
            rule_index = self.rule_iterators[self.current_rule]
            self.rule_iterators[self.current_rule] += 1
            self.current_rule = self.rules_successors[self.current_rule][rule_index]
            self.successes = 0
        return chosen_rule == self.current_rule