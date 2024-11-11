import lm_eval
import json
from pathlib import Path
from typing import List
from datetime import datetime
from lm_eval.tasks import TaskManager

playpen_eval_logger = logging.getLogger("playpen_eval_logger")


class PlaypenEvaluator:
    def __init__(self):
        pass

    def _get_executed_tasks(output_subfolder: Path, tasks: List[str]) -> (List[str], List[str]):
        executed_tasks = set()
        for json_file in output_subfolder.glob("*.json"):
            with open(json_file, "r") as file:
                try:
                    data = json.load(file)
                    if "results" in data:
                        executed_tasks.update(data["results"].keys())
                except json.JSONDecodeError:
                    playpen_eval_logger.warning(f"Warning: {json_file} could not be decoded as JSON.")

        executed_tasks = list(executed_tasks)
        pending_tasks = [task for task in tasks if task not in executed_tasks]

        return executed_tasks, pending_tasks

    def _save_reports(self) -> None:
        pass

    @staticmethod
    def list_tasks() -> None:
        pass

    @staticmethod
    def run(model_backend: str, model_args: str, tasks: List, device: str, log_samples: bool) -> None:
        # Check for already executed tasks
        executed_tasks, other_tasks = _get_executed_tasks(output_subfolder, tasks)
        task_manager = TaskManager()
        harness_tasks = task_manager.all_tasks
        pending_tasks = [t for t in other_tasks if t in harness_tasks]
        unk_tasks = [t for t in other_tasks if t not in harness_tasks]

        playpen_eval_logger.info(f"The current model has been already evaluated on the tasks: {executed_tasks}")
        playpen_eval_logger.info(f"Now evaluating on {pending_tasks}")
        playpen_eval_logger.info(f"Unknown/Not yet implemented tasks: {unk_tasks}")

        # Run evaluation for each pending task
        for task in pending_tasks:
            results = lm_eval.simple_evaluate(
                model=model,
                model_args=model_args,
                tasks=tasks,
                device=device,
                log_samples=log_samples,
            )
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            output_file_path = Path(os.path.join(output_subfolder, f"{task}_results{timestamp}.json"))
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w") as file:
                results = json.dumps(str(results))
                json.dump(results, file)

    @staticmethod
    def score() -> None:
        pass