from loguru import logger
import sys
from pathlib import Path

def setup_logger(log_dir: str = "logs", experiment_name: str = "experiment"):
    """Configure loguru logger with file and console output"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    logger.add(
        log_path / f"{experiment_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="10 days",
        compression="zip"
    )
    
    logger.info(f"Logger initialized for experiment: {experiment_name}")
    return logger
