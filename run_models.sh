#!/bin/bash

python -m eval run --model_backend hf --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct --tasks drop --device cuda:0 --trust_remote_code --results_path results
python -m eval run --model_backend hf --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct --tasks drop --device cuda:0 --trust_remote_code --results_path results
python -m eval run --model_backend hf --model_args pretrained=google/gemma-2-2b-it --tasks drop --device cuda:0 --trust_remote_code --results_path results
python -m eval run --model_backend hf --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct --tasks drop --device cuda:0 --trust_remote_code --results_path results
python -m eval run --model_backend hf --model_args pretrained=Qwen/Qwen2.5-3B-Instruct --tasks drop --device cuda:0 --trust_remote_code --results_path results
python -m eval run --model_backend hf --model_args pretrained=microsoft/Phi-3.5-mini-instruct --tasks drop --device cuda:0 --trust_remote_code --results_path results
python -m eval run --model_backend hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --tasks drop --device cuda:0 --trust_remote_code --results_path results
python -m eval run --model_backend hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct --tasks drop --device cuda:0 --trust_remote_code --results_path results
