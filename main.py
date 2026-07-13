"""
Entry point for the project.
"""
from utils.logger import get_logger
from utils.device import get_device
from utils.seed import seed_everything

def main():
    logger = get_logger("main")
    logger.info("Initializing project...")
    
    # Setup
    seed_everything(42)
    device = get_device()
    logger.info(f"Project initialized on {device}.")

if __name__ == "__main__":
    main()
