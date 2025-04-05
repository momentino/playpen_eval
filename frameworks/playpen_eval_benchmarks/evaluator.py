from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Union
from frameworks.playpen_eval_benchmarks import eval_logger
from frameworks.playpen_eval_benchmarks.tasks import CustomTaskManager
from frameworks.playpen_eval_benchmarks.tasks.utils import get_task_config, get_task
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import (
    TaskManager,
    get_task_dict,
)
from lm_eval.evaluator_utils import get_task_list

def evaluate(task: str,
             model_args:str,
             gen_kwargs:str,
             device: str,
             apply_chat_template: bool,
             log_samples: bool = False,
             model: str ="hf") -> Dict[str, Any]:
    tasks_path = str(Path(__file__).parent / "tasks")
    task_manager = CustomTaskManager(include_defaults=False, include_path=tasks_path)
    task_dict = get_task_dict(task, task_manager)

    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                if task_obj.get_config("output_type") == "generate_until":
                    if gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=gen_kwargs, update=True
                        )

                """if predict_only:
                    eval_logger.info(
                        f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                    )
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(
                            f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                        )
                    else:
                        eval_logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                        )
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (
                        default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)"""

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    # tracks all Instances/requests a model must generate output on.
    requests = defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = defaultdict(int)

    eval_tasks = get_task_list(task_dict)


    print(eval_tasks)



    """task_config = get_task_config(tasks)
    if model == "hf":
        model_args = dict(pair.split("=") for pair in model_args.split(","))
        if 'gen_kwargs' in task_config.keys():
            eval_logger.info(f"gen_kwargs is set in the task [{task}] settings. Overriding user-specified gen_kwargs if present.")
            gen_kwargs = task_config["gen_kwargs"]
        else:
            gen_kwargs = dict(pair.split("=") for pair in gen_kwargs.split(","))
        model = HFLM(device=device, gen_kwargs=gen_kwargs, **model_args)
    else:
        message = f"Model backend {model} is not supported."
        eval_logger.exception(message)
        raise Exception(message)
    task = get_task(task)
    assert task_config["output_type"] is not None
    assert task_config["output_type"] in ["generate_until_multiturn"]
    output_type = task_config["output_type"]
    if output_type == "generate_until_multiturn":
        pass
    results = task.evaluate(model, apply_chat_template)
    return results"""
