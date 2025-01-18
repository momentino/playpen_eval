from typing import Dict, Any
import torch
from frameworks.playeval_framework.tasks.utils import get_task_config, get_task
from frameworks.playeval_framework.models import HF

def evaluate(task: str, model_args:str, gen_kwargs:str, device: str, log_samples: bool = False, apply_chat_template:bool = False, model: str ="hf") -> Dict[str, Any]:
    task_config = get_task_config(task)
    if model == "hf":
        model_args = dict(pair.split("=") for pair in model_args.split(","))
        gen_kwargs = dict(pair.split("=") for pair in gen_kwargs.split(","))

        model = HF(guidance=task_config['guidance'], device=device, gen_kwargs=gen_kwargs, **model_args)
        # Since I am testing on cpu
        # model = HF(guidance=task_config['guidance'], device=device, gen_kwargs=gen_kwargs, torch_dtype=torch.float32, **model_args)
    else:
        raise Exception(f"Model backend {model} is not supported.")
    task = get_task(task, task_config)
    results = task.evaluate(model)
    return results
