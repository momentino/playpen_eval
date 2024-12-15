from typing import Dict, Any
from frameworks.playeval_framework.tasks.utils import get_task_config, get_task
from frameworks.playeval_framework.models import HF

def evaluate(task: str, model_args:str, device: str, log_samples: bool = False, apply_chat_template:bool = False, model: str ="hf") -> Dict[str, Any]:
    task_config = get_task_config(task)
    if model == "hf":
        model_args = dict(pair.split("=") for pair in model_args.split(","))
        if "max_tokens" in task_config:
            model_args['max_tokens'] = task_config['max_tokens']
        if "stop_token" in task_config:
            model_args['stop_token'] = task_config['stop_token']
        model = HF(guidance=task_config['guidance'], device=device, **model_args)
    else:
        raise Exception(f"Model backend {model} is not supported.")
    task = get_task(task)
    results = task.evaluate(model)
    return results
