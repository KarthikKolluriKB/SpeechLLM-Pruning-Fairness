import os
import logging
from logging.handlers import RotatingFileHandler


def get_logger(log_dir: str, filename: str = "train.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(os.path.join(log_dir, filename), maxBytes=10*1024*1024, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
