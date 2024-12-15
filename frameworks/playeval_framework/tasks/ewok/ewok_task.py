import torch
import numpy as np
from typing import Dict, Any
from transformers import AutoTokenizer
from tqdm import tqdm
from frameworks.playeval_framework.models import Model
from datasets import load_dataset
from frameworks.playeval_framework.tasks.task import Task

""" Taken and adapted from the original repository (https://github.com/ewok-core/ewok-paper)"""
class EwokMinimalPairsTask(Task):
    def __init__(self):
        super().__init__(task_name="ewok_minimal_pairs")
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        self.dataset = load_dataset("ewok-core/ewok-core-1.0")["test"]

    def _prepare_prompt(self, target: str, context: str = "") -> str:
        prompt = f"{context} {target}"
        return prompt

    # Adaptation from: https://github.com/benlipkin/surprisal/blob/main/surprisal/model.py, the function used in the original work of this benchmark
    def _compute_surprisal(self, model_inputs: Dict[str, torch.Tensor], logits: torch.Tensor, tokenizer: AutoTokenizer,
                           use_bos_token: bool = False) -> np.float64:
        #print(" LOGITS SHAPE ",logits.shape)
        b, n, V = logits.shape
        logits[:, :, tokenizer.pad_token_id] = -float("inf")
        logsoftmax = torch.log_softmax(logits, dim=2)
        logprobs = (
            logsoftmax[:, :-1, :]
            .gather(
                2,
                model_inputs.input_ids[:, not use_bos_token:].unsqueeze(2),
            )
            .squeeze(2)
        )
        if not use_bos_token:
            logprobs = torch.concat(((torch.ones(b, 1) * torch.nan).to(model_inputs["input_ids"].device), logprobs),
                                    dim=1)
        return -logprobs[:].cpu().float().numpy()[0]

    def _get_surprisal_bounds(self, tokenizer: AutoTokenizer, context: str, prompt: str,
                              tokenizer_max_length: int = 1024) -> (int, int):
        start = tokenizer(context,
                          padding="longest",
                          max_length=tokenizer_max_length,
                          return_tensors="pt",
                          add_special_tokens=True)["input_ids"].size()[1] if context else 0
        stop = tokenizer(prompt,
                         padding="longest",
                         max_length=tokenizer_max_length,
                         return_tensors="pt",
                         add_special_tokens=True)["input_ids"].size()[1]
        #print(" START ",start, " STOP ",stop)
        return start, stop

    def evaluate(self, model: Model) -> Dict[str, Any]:
        correct = {
            'total': [],
            'agent_properties': [],
            'social_relations': [],
            'material_dynamics': [],
            'physical_relations': [],
            'physical_dynamics': [],
            'physical_interactions': [],
            'spatial_relations': [],
            'quantitative_properties': [],
            'social_properties': [],
            'social_interactions': [],
            'material_properties': [],
        }
        for row in tqdm(self.dataset, desc="Evaluating Ewok Minimal Pairs"):
            context_1 = row["Context1"]
            context_2 = row["Context2"]
            target_1 = row["Target1"]
            target_2 = row["Target2"]

            results = {}
            results["prompt_target_1"] = {
                "target": target_1,
                "context": "",
                "prompt": self._prepare_prompt(target_1),
                "model_output": model(self._prepare_prompt(target_1))
            }
            results["prompt_target_2"] = {
                "target": target_2,
                "context": "",
                "prompt": self._prepare_prompt(target_2),
                "model_output": model(self._prepare_prompt(target_2))
            }
            results["prompt_target_1_context_1"] = {
                "target": target_1,
                "context": context_1,
                "prompt": self._prepare_prompt(target_1, context_1),
                "model_output": model(self._prepare_prompt(target_1, context_1))
            }
            results["prompt_target_1_context_2"] = {
                "target": target_1,
                "context": context_2,
                "prompt": self._prepare_prompt(target_1, context_2),
                "model_output": model(self._prepare_prompt(target_1, context_2))
            }
            results["prompt_target_2_context_1"] = {
                "target": target_2,
                "context": context_1,
                "prompt": self._prepare_prompt(target_2, context_1),
                "model_output": model(self._prepare_prompt(target_2, context_1))
            }
            results["prompt_target_2_context_2"] = {
                "target": target_2,
                "context": context_2,
                "prompt": self._prepare_prompt(target_2, context_2),
                "model_output": model(self._prepare_prompt(target_2, context_2))
            }

            try:
                tokenizer = model.get_tokenizer()
            except:
                raise Exception("No tokenizer found!")

            for key, value in results.items():
                surprisal = self._compute_surprisal(model_inputs=value["model_output"][0],
                                                    logits=value["model_output"][1], tokenizer=tokenizer)
                start, stop = self._get_surprisal_bounds(tokenizer, context=value["context"], prompt=value["prompt"])
                results[key].update(
                    {"score": -surprisal[start:stop].sum()})  # The surprisal of the target tokens given the context
            if ((results["prompt_target_1_context_1"]["score"] > results["prompt_target_1_context_2"]["score"]) and
                    (results["prompt_target_2_context_1"]["score"] < results["prompt_target_2_context_2"]["score"])):
                          score = 1
            elif ((results["prompt_target_1_context_1"]["score"] > results["prompt_target_1_context_2"]["score"]) or
                  (results["prompt_target_2_context_1"]["score"] < results["prompt_target_2_context_2"]["score"])):
                score = 0.5
            else:
                score = 0

            correct[row["Domain"].replace("-", "_")].append(score)
            correct["total"].append(score)

        formatted_results = {"model_name": model.get_model_name(),
                             "task:": self.task_name,
                             "aggregated_results": {"metric": "acc", "score": sum(correct["total"]) / len(correct["total"])},
                             "subtasks": {}
                             }
        for key, value in correct.items():
            if key != "total":
                formatted_results["subtasks"][f"{self.task_name}_{key}"] = {"metric": "acc",
                                                                            "score": sum(correct[key]) / len(
                                                                                correct[key])}
        return formatted_results

