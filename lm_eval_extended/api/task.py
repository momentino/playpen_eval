import abc
import ast
import logging
import random
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from inspect import getsource
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import datasets
import numpy as np
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api import samplers
from lm_eval.api.instance import Instance, OutputType
from lm_eval.api.metrics import bits_per_byte, mean, weighted_perplexity
from lm_eval.api.registry import (
    AGGREGATION_REGISTRY,
    DEFAULT_METRIC_REGISTRY,
    get_aggregation,
    get_metric,
    get_metric_aggregation,
    is_higher_better,
)
from lm_eval.caching.cache import load_from_cache, save_to_cache
from lm_eval.filters import build_filter_ensemble
from lm_eval.prompts import get_prompt
from lm_eval.api.task import TaskConfig, ConfigurableTask
from lm_eval.api.task import ALL_OUTPUT_TYPES

ALL_OUTPUT_TYPES.append("generate_until_multiturn")

eval_logger = logging.getLogger(__name__)


class ExtendedConfigurableTask(ConfigurableTask):
    pass



