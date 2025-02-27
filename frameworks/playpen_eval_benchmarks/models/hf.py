import torch
from typing import List, Dict
from functools import reduce
from guidance import models
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils.modeling import get_max_memory
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from frameworks.playpen_eval_benchmarks.models import Model
from frameworks.playpen_eval_benchmarks.models.guidance_chat_templates.chat_templates import CUSTOM_CHAT_TEMPLATE_CACHE

from peft import __version__ as PEFT_VERSION, PeftModel


class HF(Model):

    def __init__(self, pretrained: str,
                 device: str,
                 trust_remote_code: bool,
                 guidance: bool = False,
                 torch_dtype: str = 'auto',
                 parallelize: bool = True,
                 gen_kwargs: Dict = {},
                 peft: Optional[str] = None,
                 load_in_8bit: Optional[bool] = False,
                 load_in_4bit: Optional[bool] = False,
                 bnb_4bit_compute_dtype: Optional[str] = None
                 ) -> None:
        self.model_name = pretrained.replace("__", "/")
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
                'dtype': 'float16',
                'device_map': 'auto'
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
            if transformers.__version__ >= "4.30.0":
                if load_in_4bit:
                    if bnb_4bit_compute_dtype is not None:
                        bnb_4bit_compute_dtype = get_dtype(bnb_4bit_compute_dtype)


            self.torch_dtype = torch.float16 if torch_dtype == 'float16' else torch_dtype

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              trust_remote_code=self.trust_remote_code,
                                                              revision='main',
                                                              torch_dtype=self.torch_dtype,
                                                              device_map='auto',
                                                              bnb_4bit_compute_dtype=bnb_4bit_compute_dtype)

        if peft:
            if load_in_4bit:
                assert PEFT_VERSION >= "0.4.0", "load_in_4bit requires peft >= 0.4.0"
            self._model = PeftModel.from_pretrained(
                self._model, peft, revision=revision
            )

    def set_tokenizer_padding_side(self, padding_side: str):
        self.tokenizer.padding_side = padding_side

    def set_tokenizer_pad_token(self, pad_token: str):
        self.tokenizer.pad_token = pad_token

    def __call__(self, prompt: str) -> (torch.Tensor, torch.Tensor):
        if isinstance(self.model, PreTrainedModel):
            model_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            return model_inputs, outputs['logits']
        else:
            raise Exception('Model must be a model from Huggingface to use this method.')

    def generate(self, messages: List[Dict[str, str]] | List[str], apply_chat_template: bool):
        if isinstance(self.model, PreTrainedModel):
            if apply_chat_template:
                try:
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except:
                    # May not have a system role
                    for m in messages:
                        if m['role'] == "system":
                            m['role'] = "user"
                    messages = self._ensure_turn_taking(messages)
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = messages
            model_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.generate(
                pad_token_id=self.tokenizer.pad_token_id,
                **model_inputs,
                **self.gen_kwargs,
            )
            input = self.tokenizer.batch_decode(model_inputs['input_ids'], skip_special_tokens=True)
            text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            completions = [t[len(i):] for t, i in zip(text, input)]
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

    def get_device(self) -> str:
        return self.device
