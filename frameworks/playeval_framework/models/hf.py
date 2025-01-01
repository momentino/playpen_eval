import torch
from typing import List, Dict
from functools import reduce
from guidance import models
from accelerate import dispatch_model, infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from frameworks.playeval_framework.models import Model
from frameworks.playeval_framework.models.guidance_chat_templates.chat_templates import CUSTOM_CHAT_TEMPLATE_CACHE

class HF(Model):

    def __init__(self, pretrained: str,
                 device: str,
                 gen_kwargs: Dict,
                 trust_remote_code: bool,
                 guidance: bool = False,
                 torch_dtype: str = 'auto',
                 parallelize: bool = True,
                 ) -> None:
        self.model_name = pretrained.replace("__","/")
        super().__init__(model_name=self.model_name)
        self.gen_kwargs = gen_kwargs
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, device_map="auto")

        if self.tokenizer.pad_token is None:
            self.set_tokenizer_pad_token(self.tokenizer.eos_token)


        if guidance:
            model_config = {
                'echo': False,
                'dtype': 'float16'
            }
            self.chat_template = AutoTokenizer.from_pretrained(self.model_name).chat_template
            if self.chat_template in CUSTOM_CHAT_TEMPLATE_CACHE:
                self.model = models.Transformers(self.model_name,
                                                 chat_template=CUSTOM_CHAT_TEMPLATE_CACHE[self.chat_template],
                                                 **model_config)
            else:
                self.model = models.Transformers(self.model_name, **model_config)
            device_map = infer_auto_device_map(
                self.model.engine.model_obj,
                max_memory=None,
                no_split_module_classes=self.model.engine.model_obj._no_split_modules,
                dtype='float16'
            )
            self.model.engine.model_obj = dispatch_model(self.model.engine.model_obj, device_map=device_map)
        else:
            self.torch_dtype = torch.float16 if torch_dtype == 'float16' else torch_dtype
            print(" BEFORE MODEL ")
            model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              trust_remote_code=self.trust_remote_code,
                                                              revision='main',
                                                              torch_dtype=self.torch_dtype)
            print(" AFTER MODEL ")
            device_map = infer_auto_device_map(
                model,
                max_memory=None,
                no_split_module_classes=model._no_split_modules,
                dtype='float16'
            )
            print(" DEVICE MAP ",device_map)
            self.model = dispatch_model(model, device_map=device_map)

    def set_tokenizer_padding_side(self, padding_side:str):
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

    def generate(self,messages: List[Dict[str,str]]|List[str]):
        if isinstance(self.model, PreTrainedModel):
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)
            except:
                # May not have a system role
                for m in messages:
                    if m['role'] == "system":
                        m['role'] = "user"
                messages = self._ensure_turn_taking(messages)
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)
            outputs = self.model.generate(
                pad_token_id=self.tokenizer.pad_token_id,
                **model_inputs,
                **self.gen_kwargs,
            )
            input = self.tokenizer.batch_decode(model_inputs['input_ids'], skip_special_tokens=True)
            text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            completions = [t[len(i):] for t, i in zip(text,input)]
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

