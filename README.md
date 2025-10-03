# Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests
This work is based on the version 0.4.7 of EleutherAI's [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) project.

**DISCLAIMER**: _For any purpose that goes beyond reproducibility, it is recommended to refer to the main branch. It contains a refined version of this work._

## Cloning the repository correctly
Since it contains submodules, it's important to initialize them correctly:

```git clone --recurse-submodules https://github.com/momentino/playpen_eval.git```

## Creating the environment  
Create a new conda environment  :
```conda create triangulating```  
Define a new environment variable with (recommended if you are using models which are gated on HF)  
```conda env config vars set HF_TOKEN=<your token>```  
Run:  
```./install.sh```

## Tasks
The list of tasks used in this work is maintained in a file in the `config` folder called _task_registry_ which contains also task metadata including the sub-framework where they are implemented. 

## Evaluating a model
For evaluating models, you should use the _evaluate_ module. Arguments are mostly inspired by those in the LM Evaluation Harness, and which are mostly passed to this library's _simple_evaluate_ function afterwards.
Here are the most relevant:
- _model_backend_: the model's backend. At the moment the only supported one is Huggingface (identified with _hf_).
- _model_args_: the model name and settings, as in LM Evaluation Harness.
- _tasks_: the list of tasks on which to evaluate the model among those in the paper, separated by a space. It is possible to specify "all" to run all the tasks at once, or "remaining" to evaluate the model only on those for which a result file is not present in the results repository. The available tasks are provided in the file described in the section above.
- _parallelize_: a boolean indicating whether models should be split on multiple GPUs during inference.
- _num_fewshot_: indicates the number of few-shot examples to use during the evaluation.
- _fewshot_as_multiturn_: whether we wish to have fewshot examples presented to models in a multiturn fashion.
- _batch_size_
- _apply_chat_template_: a boolean indicating whether we want to apply the chat template to our models.
- _gen_kwargs_: generation settings as in the LM Evaluation Harness.
- _results_path_: the folder where to save results.

**Example: evaluating Qwen2.5-7B-Instruct on the WinoGrande task:**   
```python -m evaluate run --model_args pretrained=Qwen/Qwen2.5-7B-Instruct --tasks=winogrande```

## Requirements:
- python==3.10

## Citation
> @article{momentè2025triangulating,  
      title={Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests},   
      author={Filippo Momentè and Alessandro Suglia and Mario Giulianelli and Ambra Ferrari and Alexander Koller   and Oliver Lemon and David Schlangen and Raquel Fernández and Raffaella Bernardi},  
      year={2025},  
      eprint={2502.14359},  
      archivePrefix={arXiv},  
      primaryClass={cs.CL},  
      url={https://arxiv.org/abs/2502.14359},   
}  
