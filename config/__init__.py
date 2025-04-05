import os
from pathlib import Path
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
config_path = PROJECT_ROOT / "config"

LM_EVAL_TASK_LIST = ["logiqa2",
                     "cladder",
                     "eq_bench",
                     "social_iqa",
                     "glue_diagnostics",
                     "lm_pragmatics",
                     "simpletom",
                     "winogrande",
                     "mmlu_pro",
                     "bbh_fewshot",
                     "ifeval",
                     #"wm_greedy",
                     #"llm_cognitive_flexibility_original",
                     "natural_plan_5shot"]
VLLM_MAX_MODEL_LEN_BY_TASK = {
    "logiqa2": 512,
    "cladder": 512,
    "eq_bench": 512,
    "social_iqa": 512,
    "glue_diagnostics": 512,
    "lm_pragmatics": 512,
    "simpletom": 512,
    "winogrande": 512,
    "mmlu_pro": 512,
    "bbh_fewshot": 512,
    "ifeval": 512,
    #"wm_greedy": 2048,
    #"llm_cognitive_flexibility_original": 2048
    "natural_plan_5shot": 512
}