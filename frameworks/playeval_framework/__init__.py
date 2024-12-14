import logging
from pathlib import Path
from evaluator import evaluate

framework_root = Path(__file__).parent

playeval_logger = logging.getLogger('playeval')