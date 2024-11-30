from typing import Dict
from pathlib import Path
from lm_eval.tasks.planbench.planbench_eval_utils.response_evaluation import ResponseEvaluator


def process_results_plan_generation(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file = current_folder_abs / "planbench_eval_utils" / "config_files" / "logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result  =response_evaluator.evaluate_plan("plan_generation", doc, response)
    return {"acc" : result}

def process_results_plan_optimality(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_plan("plan_optimality", doc, response)
    return {"acc" : result}

def process_results_plan_reuse(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_plan("plan_reuse", doc, response)
    return {"acc" : result}

def process_results_plan_generalization(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_plan("plan_generalization", doc, response)
    return {"acc" : result}

def process_results_replanning(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_plan("replanning", doc, response)
    return {"acc" : result}

def process_results_goal_shuffling(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_plan("goal_shuffling", doc, response)
    return {"acc" : result}

def process_results_full_to_partial(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_plan("full_to_partial", doc, response)
    return {"acc" : result}

def process_results_partial_to_full(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_plan("partial_to_full", doc, response)
    return {"acc" : result}

def process_results_plan_execution(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_state(doc, response)
    return {"acc" : result}

def process_results_plan_verification(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_verification(doc, response)
    return {"acc" : result}