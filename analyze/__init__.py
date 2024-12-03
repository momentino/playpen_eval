import os
import logging
from pathlib import Path

playpen_correlation_logger = logging.getLogger("playpen_correlation_logger")

project_root = Path(os.path.abspath(__file__)).parent.parent
config_path = project_root / "config"
model_registry_path = config_path / "model_registry.yaml"
task_registry_path = config_path / "task_registry.yaml"