#!/bin/bash
pip install -e .

cd frameworks/lm_evaluation_harness
pip install -e .

cd ../playeval_framework/tasks/ewok/ewok_module
pip install -e .