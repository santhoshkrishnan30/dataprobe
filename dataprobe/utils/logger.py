# dataalchemy/utils/logger.py

import logging
from rich.logging import RichHandler
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(name: str = "dataprobe", 
                 log_file: Optional[Path] = None,
                 level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with rich formatting.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Rich console handler
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)
    
    return logger
