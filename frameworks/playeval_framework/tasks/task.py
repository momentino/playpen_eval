from typing import Dict, Any
from abc import ABC, abstractmethod
from frameworks.playeval_framework.models import Model

class Task(ABC):
    def __init__(self, task_name:str):
        self.task_name = task_name
        self.stop_token: str = None
        self.max_tokens: int = None

    @abstractmethod
    def evaluate(self, model:Model) -> Dict[str, Any]:
        pass