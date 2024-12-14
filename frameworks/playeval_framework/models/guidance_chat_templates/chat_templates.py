from guidance.chat import ChatTemplate, UnsupportedRoleException, ChatTemplateCache
from transformers import AutoTokenizer

CUSTOM_CHAT_TEMPLATE_CACHE = ChatTemplateCache()

# --------------------------------------------------
# @@@@ Llama3.2 @@@@
# --------------------------------------------------

llama32_template = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct').chat_template
class Llama32Template(ChatTemplate):
    template_str = llama32_template
    def get_role_start(self, role_name):
        if role_name == "system":
            return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        elif role_name == "user":
            return "<|start_header_id|>user<|end_header_id|>\n\n"
        elif role_name == "assistant":
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        return "<|eot_id|>"

CUSTOM_CHAT_TEMPLATE_CACHE[llama32_template] = Llama32Template

# --------------------------------------------------
# @@@@ Phi-3.5 @@@@
# --------------------------------------------------

phi35_template = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct').chat_template
class Phi35Template(ChatTemplate):
    template_str = phi35_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return "<|user|>"
        elif role_name == "assistant":
            return "<|assistant|>"
        elif role_name == "system":
            return "<|system|>"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        return "<|end|>"

CUSTOM_CHAT_TEMPLATE_CACHE[phi35_template] = Phi35Template

# --------------------------------------------------
# @@@@ Qwen2.5 @@@@
# --------------------------------------------------

qwen25_template = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct').chat_template
class Qwen25Template(ChatTemplate):
    template_str = qwen25_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return "<|im_start|>user\n"
        elif role_name == "assistant":
            return "<|im_start|>assistant\n"
        elif role_name == "system":
            return "<|im_start|>system\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        return "<|im_end|>\n"

CUSTOM_CHAT_TEMPLATE_CACHE[qwen25_template] = Qwen25Template