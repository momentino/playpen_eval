import json
from typing import List, Dict
from pathlib import Path
from frameworks.playpen_eval_benchmarks.tasks.llm_cognitive_flexibility.wcst_test import WCSTConfig, WCST
from frameworks.playpen_eval_benchmarks.tasks.llm_cognitive_flexibility.lnt_test import LNTConfig, LNT

def wcst_instance_generation(num_evaluations: int, trials: int) -> List[List[Dict]]:
    data = []
    config: WCSTConfig = WCSTConfig()
    for n in range(num_evaluations):
        test = WCST(config)
        instances = []
        for t in range(trials):
            card = test.deck[t]
            options = test.generate_options(card)
            instance = {'card': card, 'options': options}
            instances.append(instance)
        data.append(instances)
    return data

def lnt_instance_generation(num_evaluations: int, trials: int) -> List[List[Dict]]:
    data = []
    config: LNTConfig = LNTConfig()
    for n in range(num_evaluations):
        test = LNT(config)
        instances = []
        for t in range(trials):
            sequence = test.generate_sequence()
            instance = {'sequence':sequence}
            instances.append(instance)
        data.append(instances)
    return data

if __name__ == "__main__":
    num_evaluations = 8
    trials = 25
    experiment = "lnt" #change the name into 'lnt' for generating data for that experiment
    output_path = Path(__file__).parent / "revisited_data" / experiment
    output_path.mkdir(parents=True, exist_ok=True)
    if experiment == 'wcst':
        data = wcst_instance_generation(num_evaluations, trials)
    elif experiment == 'lnt':
        data = lnt_instance_generation(num_evaluations, trials)
    else:
        raise Exception("Experiment unknown in the LLM-Cognitive-Flexibility dataset's data generation script.")
    json.dump(data, open(output_path / "data.json", 'w'))