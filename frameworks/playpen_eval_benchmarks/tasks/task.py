from typing import Dict, Any
from abc import ABC, abstractmethod
from frameworks.playpen_eval_benchmarks.models import Model

class Task(ABC):
    def __init__(self, task_name:str):
        self.task_name = task_name

    @abstractmethod
    def evaluate(self, model:Model, apply_chat_template: bool) -> Dict[str, Any]:
        pass

class InteractiveTask(Task):
    def __init__(self, task_name: str, runs: int = 1, ):
        super().__init__(task_name)


    @abstractmethod
    def evaluate(self, model:Model, apply_chat_template: bool) -> Dict[str, Any]:
        pass
