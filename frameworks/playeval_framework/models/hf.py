from guidance import models
from transformers import AutoModelForCausalLM
from frameworks.playeval_framework.models import Model
class HF(Model):

    def __init__(self, pretrained: str, device: str, trust_remote_code: bool, guidance: bool = False) -> None:
        model_name = pretrained.replace("/","__")
        super().__init__(model_name=model_name)
        if guidance:
            self.model =  models.Transformers(pretrained, device=device)
        else:
            # More info?
            self.model = AutoModelForCausalLM.from_pretrained(pretrained,
                                                              trust_remote_code=trust_remote_code,
                                                              revision='main',
                                                              device=device)

    """def generate_until(self, messages: List[Dict],
                          return_full_text: bool = False,
                          log_messages: bool = False) -> Tuple[Any, Any, str]:
        pass"""
