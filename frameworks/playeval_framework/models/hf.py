import torch
from guidance import models
from typing import List
from functools import reduce
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from frameworks.playeval_framework.models import Model
from frameworks.playeval_framework.models.guidance_chat_templates.chat_templates import CUSTOM_CHAT_TEMPLATE_CACHE


class HF(Model):

    def __init__(self, pretrained: str,
                 device: str,
                 trust_remote_code: bool,
                 guidance: bool = False,
                 stop_token: str = None,
                 max_tokens: int = None,
                 torch_dtype: str = 'auto') -> None:
        self.model_name = pretrained.replace("__","/")
        super().__init__(model_name=self.model_name)
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if guidance:
            model_config= {
                'echo': False,
                'device_map': device,
            }
            self.chat_template = AutoTokenizer.from_pretrained(self.model_name).chat_template
            if self.chat_template in CUSTOM_CHAT_TEMPLATE_CACHE:
                self.model =  models.Transformers(self.model_name, chat_template=CUSTOM_CHAT_TEMPLATE_CACHE[self.chat_template], **model_config)
            else:
                self.model = models.Transformers(self.model_name,  **model_config)
        else:
            self.torch_dtype = torch.float16 if torch_dtype == 'float16' else torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              trust_remote_code=self.trust_remote_code,
                                                              revision='main',
                                                              torch_dtype=self.torch_dtype)
            self.model.to(self.device)

    def __call__(self, prompt: str) -> (torch.Tensor, torch.Tensor):
        if isinstance(self.model, PreTrainedModel):
            # TODO: Need to check if it works for non-chat models
            #messages = [{"role": "user", "content": prompt}]
            #text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text=prompt
            model_inputs = self.tokenizer([text], return_tensors="pt", padding="longest", max_length=1024, add_special_tokens=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            return model_inputs, outputs['logits']
        else:
            raise Exception('Model must be a model from Huggingface to use this method.')

    def generate(self):
        if isinstance(self.model, PreTrainedModel):
            pass
        else:
            raise Exception('Model must be a model from Huggingface to use this method.')

    def generate_guidance(self, prompt:List) -> str:
        if isinstance(self.model, models.Transformers):
            lm = reduce(lambda acc, p: acc + p, prompt, self.model)
            return lm
        else:
            raise Exception('Model must be a Transformer from Guidance to use this method.')

    def get_tokenizer(self):
        return self.tokenizer

