from typing import Dict, Any
from frameworks.playpen_eval_benchmarks import eval_logger
from frameworks.playpen_eval_benchmarks.tasks.utils import get_task_config, get_task
from frameworks.playpen_eval_benchmarks.models import HF

def evaluate(task: str, model_args:str, gen_kwargs:str, device: str, apply_chat_template: bool, log_samples: bool = False, model: str ="hf") -> Dict[str, Any]:
    task_config = get_task_config(task)
    if model == "hf":
        model_args = dict(pair.split("=") for pair in model_args.split(","))
        if 'gen_kwargs' in task_config.keys():
            eval_logger.info(f"gen_kwargs is set in the task [{task}] settings. Overriding user-specified gen_kwargs if present.")
            gen_kwargs = task_config["gen_kwargs"]
        else:
            gen_kwargs = dict(pair.split("=") for pair in gen_kwargs.split(","))
        model = HF(device=device, gen_kwargs=gen_kwargs, **model_args)
    else:
        message = f"Model backend {model} is not supported."
        eval_logger.exception(message)
        raise Exception(message)
    task = get_task(task)
    print(" READY FOR EVALUATION ",task)
    results = task.evaluate(model, apply_chat_template)
    return results
