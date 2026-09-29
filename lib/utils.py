import torch
import numpy as np
import random
import sys
import logging


# -----------------------------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)  # build-in random module
    np.random.seed(seed)  # numpy random module
    torch.manual_seed(seed)  # PyTorch seed for CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch seed for all GPU devices (if using CUDA)
    torch.backends.cudnn.deterministic = True  # disabling CuDNN's non-deterministic optimizations
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------------------------
class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# -----------------------------------------------------------------------------------------------
def setup_logger(name: str, log_dir: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"{log_dir}/{name}.log", mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.propagate = False

    return logger
