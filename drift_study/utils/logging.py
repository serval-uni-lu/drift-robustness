import logging.config
import warnings
from typing import Any, Dict

from optuna.exceptions import ExperimentalWarning


def configure_logger(config: Dict[str, Any]):
    logging_config = config.get("logging")
    if logging_config is not None:
        logging.config.dictConfig(logging_config)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
