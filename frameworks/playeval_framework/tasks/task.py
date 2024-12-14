from typing import Dict, Any
from abc import ABC, abstractmethod

class Task(ABC):
    def __init__(self, task_name:str):
        self.task_name = task_name

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        pass