from guidance import models
from typing import List
from functools import reduce

from guidance.chat import ChatTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llava_onevision.convert_llava_onevision_weights_to_hf import chat_template

from frameworks.playeval_framework.models import Model
from frameworks.playeval_framework.models.guidance_chat_templates.chat_templates import CUSTOM_CHAT_TEMPLATE_CACHE


class HF(Model):

    def __init__(self, pretrained: str, device: str, trust_remote_code: bool, guidance: bool = False) -> None:
        model_name = pretrained.replace("/","__")
        super().__init__(model_name=model_name)
        if guidance:
            model_config= {
                'echo': False,
                'device_map': device,
            }
            chat_template = AutoTokenizer.from_pretrained(pretrained).chat_template
            if chat_template in CUSTOM_CHAT_TEMPLATE_CACHE:
                self.model =  models.Transformers(pretrained, chat_template=CUSTOM_CHAT_TEMPLATE_CACHE[chat_template], **model_config)
            else:
                self.model = models.Transformers(pretrained,  **model_config)
        else:
            # More info?
            self.model = AutoModelForCausalLM.from_pretrained(pretrained,
                                                              trust_remote_code=trust_remote_code,
                                                              revision='main',
                                                              device=device)

    def generate(self):
        pass

    def generate_guidance(self, prompt:List) -> str:
        lm = reduce(lambda acc, p: acc + p, prompt, self.model)
        return lm
