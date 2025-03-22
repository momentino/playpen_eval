import torch
import transformers
from typing import List, Dict, Optional, Union
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, BitsAndBytesConfig
from frameworks.playpen_eval_benchmarks.models import Model
from peft import __version__ as PEFT_VERSION, PeftModel


class HF(Model):

    def __init__(self, pretrained: str,
                 device: str,
                 trust_remote_code: bool,
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

        model_kwargs = {}
        load_in_4bit = bool(load_in_4bit)
        load_in_8bit = bool(load_in_8bit)
        if transformers.__version__ >= "4.30.0":
            if load_in_4bit:
                if bnb_4bit_compute_dtype is not None:
                    bnb_4bit_compute_dtype = self.get_dtype(bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
            )
            model_kwargs['quantization_config'] = bnb_config
        self.torch_dtype = torch.float16 if torch_dtype == 'float16' else torch_dtype



        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          trust_remote_code=self.trust_remote_code,
                                                          revision='main',
                                                          torch_dtype=self.torch_dtype,
                                                          device_map='auto',
                                                          **model_kwargs)
        if peft:
            if load_in_4bit:
                if version.parse(PEFT_VERSION) < version.parse("0.4.0"):
                    raise AssertionError("load_in_4bit requires peft >= 0.4.0")
            self.model = PeftModel.from_pretrained(
                self.model, peft, revision="main", low_cpu_mem_usage=True, torch_dtype=self.torch_dtype,
                ephemeral_gpu_offload=True
            ) #device_map='balanced_low_0',

        print("PEFT ", self.model)
        print(" TORCH DTYPE ", self.torch_dtype)
        print(" Memory summary ", torch.cuda.memory_summary())

    def set_tokenizer_padding_side(self, padding_side: str):
        self.tokenizer.padding_side = padding_side

    def set_tokenizer_pad_token(self, pad_token: str):
        self.tokenizer.pad_token = pad_token

    # From eval harness
    def get_dtype(self, dtype: Union[str, torch.dtype]) -> torch.dtype:
        """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
        if isinstance(dtype, str) and dtype != "auto":
            # Convert `str` args torch dtype: `float16` -> `torch.float16`
            _torch_dtype = getattr(torch, dtype)
        else:
            _torch_dtype = dtype
        return _torch_dtype

    def __call__(self, prompt: str) -> (torch.Tensor, torch.Tensor):
        if isinstance(self.model, Union[PreTrainedModel,PeftModel]):
            model_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            return model_inputs, outputs['logits']
        else:
            raise Exception('Model must be a model from Huggingface to use this method.')

    def generate(self, messages: List[Dict[str, str]] | List[str], apply_chat_template: bool):
        print(" ENTRO ",len(messages))
        if isinstance(self.model, Union[PreTrainedModel,PeftModel]):
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

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self) -> str:
        return self.device
