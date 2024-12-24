import torch
from typing import List
from functools import reduce
from guidance import models
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from frameworks.playeval_framework.models import Model
from frameworks.playeval_framework.models.guidance_chat_templates.chat_templates import CUSTOM_CHAT_TEMPLATE_CACHE


class HF(Model):

    def __init__(self, pretrained: str,
                 device: str,
                 trust_remote_code: bool,
                 guidance: bool = False,
                 torch_dtype: str = 'auto') -> None:
        self.model_name = pretrained.replace("__","/")
        super().__init__(model_name=self.model_name)
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if guidance:
            model_config = {
                'echo': False,
                'device_map': device,
            }
            self.chat_template = AutoTokenizer.from_pretrained(self.model_name).chat_template
            if self.chat_template in CUSTOM_CHAT_TEMPLATE_CACHE:
                self.model = models.Transformers(self.model_name,
                                                 chat_template=CUSTOM_CHAT_TEMPLATE_CACHE[self.chat_template],
                                                 **model_config)
            else:
                self.model = models.Transformers(self.model_name, **model_config)
        else:
            self.torch_dtype = torch.float16 if torch_dtype == 'float16' else torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              trust_remote_code=self.trust_remote_code,
                                                              revision='main',
                                                              torch_dtype=self.torch_dtype)
            self.model.to(self.device)

    def set_tokenizer_padding_side(self, padding_side:str, ):
        self.tokenizer.padding_side = padding_side

    def set_tokenizer_pad_token(self, pad_token:str):
        self.tokenizer.pad_token = pad_token

    def __call__(self, prompt: str) -> (torch.Tensor, torch.Tensor):
        if isinstance(self.model, PreTrainedModel):
            model_inputs = self.tokenizer([prompt], return_tensors="pt", padding="longest", max_length=1024, add_special_tokens=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            return model_inputs, outputs['logits']
        else:
            raise Exception('Model must be a model from Huggingface to use this method.')

    def generate(self,prompt: str, system:str = None, **kwargs):
        if isinstance(self.model, PreTrainedModel):
            if self.tokenizer.chat_template is not None:
                print(self.tokenizer.chat_template)
                messages = []
                if system is not None:
                    messages.append({"role":"system", "content": system})
                messages.append({"role": "user", "content": prompt})
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self.model.generate(
                **model_inputs,
                **kwargs,
            )
            text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            completions = [t[len(p):] for t, p in zip(text, [prompt])]
            return completions
        else:
            raise Exception('Model must be a model from Huggingface to use this method.')

    def generate_guidance(self, prompt: List) -> str:
        if isinstance(self.model, models.Transformers):
            lm = reduce(lambda acc, p: acc + p, prompt, self.model)
            return lm
        else:
            raise Exception('Model must be a Transformer from Guidance to use this method.')

    def get_tokenizer(self):
        return self.tokenizer

