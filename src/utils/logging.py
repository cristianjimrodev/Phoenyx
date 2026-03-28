import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", log_file: str = "logs/trading.log",
                  rotation: str = "10 MB", retention: str = "30 days"):
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, format=fmt, level=level, rotation=rotation, retention=retention)

    return logger
