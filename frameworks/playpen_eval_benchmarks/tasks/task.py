from typing import Dict, Any
from abc import ABC, abstractmethod
from frameworks.playpen_eval_benchmarks.models import Model

class Task(ABC):
    def __init__(self, task_name:str):
        self.task_name = task_name
        self.stop_token: str = None
        self.max_tokens: int = None
        self.max_new_tokens: int = None

    @abstractmethod
    def evaluate(self, model:Model, apply_chat_template: bool) -> Dict[str, Any]:
        pass