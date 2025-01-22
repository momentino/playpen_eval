from typing import Dict, List
from abc import ABC, abstractmethod
class Model(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def __call__(self, prompt):
        pass

    @abstractmethod
    def generate(self,messages:List[Dict[str,str]]):
        pass

    def get_model_name(self) -> str:
        return self.model_name

    def _ensure_turn_taking(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return []

        new_messages = [messages[0]]
        for m in messages[1:]:
            if m['role'] == new_messages[-1]['role']:
                new_messages[-1]['content'] += m['content']
            else:
                new_messages.append(m)
        return new_messages