import os
import json
import yaml
import logging
import lm_eval
from pathlib import Path
from typing import List, Dict, Any
from config import project_root

BANNER = \
"""
 .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .-----------------.
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |   ______     | || |   _____      | || |      __      | || |  ____  ____  | || |   ______     | || |  _________   | || | ____  _____  | |
| |  |_   __ \   | || |  |_   _|     | || |     /  \     | || | |_  _||_  _| | || |  |_   __ \   | || | |_   ___  |  | || ||_   \|_   _| | |
| |    | |__) |  | || |    | |       | || |    / /\ \    | || |   \ \  / /   | || |    | |__) |  | || |   | |_  \_|  | || |  |   \ | |   | |
| |    |  ___/   | || |    | |   _   | || |   / ____ \   | || |    \ \/ /    | || |    |  ___/   | || |   |  _|  _   | || |  | |\ \| |   | |
| |   _| |_      | || |   _| |__/ |  | || | _/ /    \ \_ | || |    _|  |_    | || |   _| |_      | || |  _| |___/ |  | || | _| |_\   |_  | |
| |  |_____|     | || |  |________|  | || ||____|  |____|| || |   |______|   | || |  |_____|     | || | |_________|  | || ||_____|\____| | |
| |              | || |              | || |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 
 .----------------.  .----------------.  .----------------.  .----------------.                                                             
| .--------------. || .--------------. || .--------------. || .--------------. |                                                            
| |  _________   | || | ____   ____  | || |      __      | || |   _____      | |                                                            
| | |_   ___  |  | || ||_  _| |_  _| | || |     /  \     | || |  |_   _|     | |                                                            
| |   | |_  \_|  | || |  \ \   / /   | || |    / /\ \    | || |    | |       | |                                                            
| |   |  _|  _   | || |   \ \ / /    | || |   / ____ \   | || |    | |   _   | |                                                            
| |  _| |___/ |  | || |    \ ' /     | || | _/ /    \ \_ | || |   _| |__/ |  | |                                                            
| | |_________|  | || |     \_/      | || ||____|  |____|| || |  |________|  | |                                                            
| |              | || |              | || |              | || |              | |                                                            
| '--------------' || '--------------' || '--------------' || '--------------' |                                                            
 '----------------'  '----------------'  '----------------'  '----------------'                                                             
"""

print(BANNER)

def configure_logging(project_root):
    # Configure logging
    with open(os.path.join(project_root, "logging.yaml")) as f:
        conf = yaml.safe_load(f)
        log_fn = conf["handlers"]["file_handler"]["filename"]
        log_fn = os.path.join(project_root, log_fn)
        conf["handlers"]["file_handler"]["filename"] = log_fn
        logging.config.dictConfig(conf)


def get_logger(name):
    return logging.getLogger(name)

def get_executed_tasks(model_results_path: Path, tasks: List[str]) -> (List[str], List[str]):
    executed_tasks = set()
    for json_file in model_results_path.glob("*.json"):
        with open(json_file, "r") as file:
            try:
                data = json.load(file)
                if "task_results" in data:
                    executed_tasks.update(data["task_results"].keys())
            except json.JSONDecodeError:
                print(f"Warning: {json_file} could not be decoded as JSON.")

    executed_tasks = list(executed_tasks)
    pending_tasks = [task for task in tasks if task not in executed_tasks]

    return executed_tasks, pending_tasks