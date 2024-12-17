#!/bin/bash
pip install requirements.txt

cd frameworks/lm_evaluation_harness
pip install requirements.txt

cd ../frameworks/playeval_framework/tasks/ewok/ewok_module
pip install -e .