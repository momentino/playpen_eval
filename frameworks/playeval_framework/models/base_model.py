from abc import ABC, abstractmethod
class Model(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def __call__(self, prompt):
        pass

    @abstractmethod
    def generate(self,prompt: str, system:str = None, **kwargs):
        pass

    def get_model_name(self) -> str:
        return self.model_name