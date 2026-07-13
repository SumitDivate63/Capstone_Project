"""Clean logging system."""
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler('outputs/logs/project.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
