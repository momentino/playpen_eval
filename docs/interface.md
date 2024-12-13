# Evaluating models
In order to evaluate a model using our pipeline, you have to run the ``` python -m eval run ``` script.

The script supports several arguments:

* ```--model_backend```: the backend supporting the model (e.g. _hf_ for HuggingFace).
* ```--model_args```: settings to pass to pass to the model constructor, comma separated. If you just wish to specify a model name for the ```hf``` backend, you may type ```pretrained=<model_name>```.
* ```--tasks```: the tasks on which to run the evaluation. It can be a list of tasks e.g. ```planbench fantom_full ...```, or you can type ```all``` to be evaluated on each task.
* ```--device```: the device on which to run the evaluation.
* ```--trust_remote_code```: whether to trust remote code or not.
* ```--results_path```: folder where to save results.

### Evaluating all the benchmarks
Here the model is _allenai/Llama-3.1-Tulu-3-8B_ and the backend is Huggingface (_hf_)
```bash
python -m eval run 
--model_backend hf 
--model_args pretrained=allenai/Llama-3.1-Tulu-3-8B 
--tasks all 
--device cuda:0 
--trust_remote_code 
--results_path results
```
### Evaluating on a specific task
Here the model is _allenai/Llama-3.1-Tulu-3-8B_ and the backend is Huggingface (_hf_). We are evaluating only on _planbench_
```bash
python -m eval run 
--model_backend hf 
--model_args pretrained=allenai/Llama-3.1-Tulu-3-8B 
--tasks planbench 
--device cuda:0 
--trust_remote_code 
--results_path results
```
# Running the correlation analysis experiments

#TODO