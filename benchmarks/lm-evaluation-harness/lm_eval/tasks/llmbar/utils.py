import re

def reverse_label(label):
    if label is None:
        return None
    elif label == "1":
        return "2"
    elif label == "2":
        return "1"
    else:
        raise NotImplementedError

def parse(result):
    completion = result[0]
    parsing = {
        "1": "(?:^|\\n) ?Output \\(a\\)",
        "2": "(?:^|\\n) ?Output \\(b\\)"
    }
    winner = None
    for return_value, exp in parsing.items():
        if (re.compile(exp)).search(completion):
            winner = return_value
    return winner

def process_results(doc, result):
    winner = parse(result)
    label = doc["label"]
    correct = 0
    if winner is not None and str(winner) == str(label):
        correct = 1
    return {"acc": correct}

def process_results_swapped(doc, result):
    winner = parse(result)
    label = reverse_label(str(doc["label"]))
    correct = 0
    if winner is not None and str(winner) == str(label):
        correct = 1
    return {"acc": correct}