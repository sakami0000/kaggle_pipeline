from logging import Logger, StreamHandler, Formatter, FileHandler, getLogger
import os
from pathlib import Path
import random

import numpy as np
import torch


def set_seed(seed: int = 1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(logger: Logger, log_file_path: str):
    handler = StreamHandler()
    handler.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler(log_file_path, mode='w')
    handler.setLevel('DEBUG')
    logger.addHandler(handler)
