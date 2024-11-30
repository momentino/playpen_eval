""" Parts of the code here is taken and adapted from the original repository for this benchmark https://github.com/aviaefrat/lmentry/tree/main"""

import re
import string
from num2words import num2words
from typing import List

""" Taken from: https://github.com/aviaefrat/lmentry/tree/main """
word_regex_template = r'(The word )?["\'\.,]*\b{}\b["\'\.,]*'
sentence_regex_template = r'((The sentence )?["\'\.,]*{}["\'\.,]*|The sentence)'
letter_regex_template = r'(The letter )?["\'\.,]*\b{}\b["\'\.,]*'
number_regex_template = r'(The number )?["\'\.,]*\b{}\b["\'\.,]*'

def standardized_number_regex(number: int):
    digits_string = str(number)
    word_form = num2words(number)
    return fr'(\b{digits_string}\b|{word_form}|{word_form.replace("-", " ")})'


def the_number_regex(number: int):
    standardized_number_regex_ = standardized_number_regex(number)
    return number_regex_template.format(standardized_number_regex_)


def the_word_regex(word: str):
    return word_regex_template.format(word)


def the_words_regex(words: List[str]):
    # we use `[^a-zA-Z0-9_]` instead of `\W` as we always lowercase the pattern in
    # `certainty_scorer`. See the to-do suggestion in normalize_string
    W = r"[^a-zA-Z0-9_]"

    words_regex = rf"{W}*"
    for i, word in enumerate(words):
        words_regex += rf"{word}{W}+"
        if i == len(words) - 2:
            words_regex += rf"(and{W}+)?"
    words_regex += rf"{W}*"

    return rf"(The words )?{words_regex}"


def the_list_regex(words: List[str]):

    words_as_string = '", "'.join(words)
    words_as_string = rf'\["{words_as_string}"]'

    return rf"(((The list )?{words_as_string})|the list)"


def the_sentence_regex(sentence: str):
    return sentence_regex_template.format(sentence)


def the_letter_regex(letter: str):
    return letter_regex_template.format(letter)

def normalize_string(s: str):
    s = s.strip()
    s = s.lower()
    s = re.sub(r" +", " ", s)

    return s


def normalize_prediction(prediction: str, truncate_prediction: bool = False):
    prediction = normalize_string(prediction)
    prediction = prediction.replace("_", " ")  # this is an easy hack to not have \w include "_"

    if truncate_prediction:
        prediction = prediction.split("\n")[0]
        prediction = prediction.split(".")[0]

    return prediction


def _simple_scorer(prediction, answer, allow_answer_pattern_repetitions=True) -> dict:
    score = 0

    if re.match(rf"{answer}\.?$", prediction, flags=re.IGNORECASE):
        score = 1

    if allow_answer_pattern_repetitions:
        alphanumeric_pattern = r"\b[a-z\d]+\b"
        all_alphanumeric_words = re.findall(alphanumeric_pattern, prediction)
        if all([re.match(answer + "$", word) for word in all_alphanumeric_words]):
            score = 1

    return {"acc": score}


def process_results_ends_with_letter(doc, results) -> dict:
    prediction = normalize_prediction(results[0])

    letter = doc["letter"]

    before_the_letter = r"\w*" if letter in {"a", "i"} else r"\w+"

    word = rf"{before_the_letter}{letter}"
    answer = the_word_regex(word)

    return _simple_scorer(prediction, answer)

def process_results_starts_with_letter(doc, results) -> dict:
    prediction = normalize_prediction(results[0])

    letter = doc["letter"]
    after_the_letter = r"\w*" if letter in {"a", "i"} else r"\w+"

    word = rf"{letter}-?{after_the_letter}"
    answer = the_word_regex(word)

    return _simple_scorer(prediction, answer)

def process_results_ends_with_word(doc, results) -> dict:
    prediction = normalize_prediction(results[0])
    word = the_word_regex(doc["word"])
    valid_word = r"(a|i|\w\w+)"
    allowed_sentence = rf".*{valid_word}[^a-zA-Z0-9_]+{word}[^a-zA-Z0-9_]*"

    return _simple_scorer(prediction, allowed_sentence)

def process_results_starts_with_word(doc, results) -> dict:
    pred = normalize_prediction(results[0])
    word = the_word_regex(doc["word"])
    valid_word = r"(a|i|\w\w+)"
    allowed_sentence = rf"[^a-zA-Z0-9_]*{word}[^a-zA-Z0-9_]+{valid_word}.*"

    return _simple_scorer(pred, allowed_sentence)

def process_results_first_last_letter(doc, results) -> dict:
    answer = the_letter_regex(doc["answer"])
    prediction = results[0]
    prediction = normalize_prediction(prediction)

    return _simple_scorer(prediction, answer)

def process_results_first_last_word(doc, results) -> dict:
    answer = the_word_regex(doc["answer"])
    prediction = results[0]
    prediction = normalize_prediction(prediction)

    return _simple_scorer(prediction, answer)

def process_results_sentence_containing(doc, results) -> dict:
    word = doc["word"].lower()
    prediction = results[0]
    prediction = normalize_prediction(prediction)
    allowed_sentence = rf"((.*\b{word}\b.+)|(.+\b{word}\b.*))"
    allowed_sentence = the_sentence_regex(allowed_sentence)

    if re.search(allowed_sentence, prediction):
        score = 1
    else:
        score = 0
    return {"acc":score}

def process_results_sentence_not_containing(doc, results) -> dict:
    word = doc["word"].lower()
    prediction = results[0]
    prediction = normalize_prediction(prediction)
    if re.search(f'\\b{word}\\b', prediction):
        score = 0
    else:
        prediction_words = re.findall(r"\b[a-z]+\b", prediction)
        if len(prediction_words) == 0:
            score = 0
        elif len(prediction_words) == 1:
            score = 0

        else:
            if word in prediction:
                score = 1
            else:
                score = 1
    return {"acc": score}

def process_results_word_before_after(doc, results) -> dict:
    prediction = normalize_prediction(results[0])

    answer = doc["answer"]
    answer = answer.lower()
    answer = the_word_regex(answer)

    return _simple_scorer(prediction, answer)

def process_results_word_containing(doc, results) -> dict:
    prediction = normalize_prediction(results[0])

    letter = doc["letter"]

    word = (rf"\w*{letter}\w*"
            if letter in {"a", "i"}
            else rf"((\w+{letter}\w*)|(\w*{letter}\w+)|({letter}-\w+))"
            # the third option under `else` is for cases like "e-mail" and "x-ray"
            )
    answer = the_word_regex(word)

    return _simple_scorer(prediction, answer)

def process_results_word_not_containing(doc, results) -> dict:
    prediction = normalize_prediction(results[0])

    letter = doc["letter"]

    input_ = doc["input"]

    allowed_letters = string.ascii_lowercase.replace(letter, "")
    allowed_word = rf"(\b[{allowed_letters}]{{2,}}\b)"
    if letter != "a":
        allowed_word += r"|(a)"
    if letter != "i":
        allowed_word += r"|(i)"
    if letter != "a" or letter != "i":
        allowed_word = "(" + allowed_word + ")"

    allowed_word = the_word_regex(allowed_word)

    return _simple_scorer(prediction, allowed_word, allow_answer_pattern_repetitions=False)

""" End taken from https://github.com/aviaefrat/lmentry/tree/main"""

def doc_to_target_all_words_from_category(doc) -> int:
    target = 0 if len(doc["distractors"]) > 0 else 1
    return target

def doc_to_target_any_words_from_category(doc) -> int:
    target = 0 if len(doc["category_words"]) == 0 else 1
    return target


