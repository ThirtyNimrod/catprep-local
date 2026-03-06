import logging
import os
from pathlib import Path

# Ensure logs directory exists at the root level
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = LOGS_DIR / "app.log"

os.makedirs(LOGS_DIR, exist_ok=True)

# Central log formatter
FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Suppress noisy third-party loggers that pollute the terminal
for _noisy in ("pydantic", "httpx", "httpcore", "urllib3", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Returns a configured logger with the given name.

    Parameters
    ----------
    name     : Logger name (used in the log prefix and as the logger identity).
    level    : Logging level (default INFO).
    log_file : Path to the output log file. Defaults to logs/app.log.
               Pass an explicit path to write to a separate file
               (e.g. logs/build_knowledge_graph.log).
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(level)
        target_file = log_file if log_file is not None else DEFAULT_LOG_FILE

        # File Handler
        file_handler = logging.FileHandler(target_file, encoding="utf-8")
        file_handler.setFormatter(FORMATTER)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(FORMATTER)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Prevent log messages from propagating to the root logger twice
        logger.propagate = False

    return logger
