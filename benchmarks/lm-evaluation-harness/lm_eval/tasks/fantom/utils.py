import re
from collections import Counter

def process_results_belief_choice(doc, response) -> dict:
    model_response = response[0]
    int_to_alphabet = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    answer = int_to_alphabet[doc['correct_answer']]
    right_guess = model_response.startswith("(" + answer + ")") or model_response.startswith(answer + ")") or model_response.startswith(answer + ".") or model_response.startswith(answer + ":") or model_response.startswith(answer + ",") or "({})".format(answer) in model_response or answer == model_response
    return {'acc': right_guess}

def process_results_list(doc, response) -> dict:
    model_response = response[0]
    excluded_aware_character = False
    included_unaware_character = False
    for character in doc['correct_answer']:
        if character.lower() not in model_response.lower():
            excluded_aware_character = True
            break

    for character in doc['wrong_answer']:
        if character.lower() in model_response.lower():
            included_unaware_character = True
            break
    return {'acc': not(excluded_aware_character or included_unaware_character)}

def process_results_binary(doc, response) -> dict:
    model_response = response[0].lower().strip("'").strip('"')
    result = -1
    if " yes," in model_response or " yes " in model_response or model_response.startswith(
            "yes") or " yes." in model_response or " knows " in model_response or model_response.lower().startswith("true"):
        result = 1
    elif " no," in model_response or " no " in model_response or model_response.startswith(
            "no") or " no." in model_response or " does not know " in model_response or " doesn't know " in model_response or model_response.lower().startswith(
            "false"):
        result = 0
    match = result == doc_to_target_binary(doc)
    return {'exact_match': match}


def process_results_f1(doc, response) -> dict:
    pred, gold = response, doc['correct_answer']
    f1 = token_f1(pred, gold)
    return {'f1': f1}

def token_f1(pred, gold) -> float:
    model_response = pred[0].split()
    ground_truth = gold.split()
    common = Counter(ground_truth) & Counter(model_response)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(model_response)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_choices_list(doc) -> list:
    correct_answers = doc['correct_answer']
    wrong_answers = doc['wrong_answer']
    choices = []
    choices.extend(correct_answers)
    choices.extend(wrong_answers)
    return choices

def doc_to_target_binary(doc) -> int:
    target = 1 if doc['correct_answer'] == 'yes' else 0
    return target

def doc_to_choices(doc) -> list:
    input_text = doc['input_text']
    pattern = r'\(a\)(.*?)\(b\)'
    match = re.search(pattern, input_text, re.DOTALL)  # re.DOTALL allows . to match newline characters
    if match:
        optionA = match.group(1).strip()  # Return the extracted text, stripped of leading/trailing whitespace
    else:
        optionA = None  # Return None if no match is found

    pattern = r'\(b\)(.*?)Choose an answer from above:'
    match = re.search(pattern, input_text, re.DOTALL)  # re.DOTALL allows . to match newline characters
    if match:
        optionB = match.group(1).strip()  # Return the extracted text, stripped of leading/trailing whitespace
    else:
        optionB = None  # Return None if no match is found
    return [optionA, optionB]
