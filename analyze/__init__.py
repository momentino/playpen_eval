import os
import logging
from pathlib import Path

playpen_correlation_logger = logging.getLogger("playpen_correlation_logger")

project_root = Path(os.path.abspath(__file__)).parent.parent