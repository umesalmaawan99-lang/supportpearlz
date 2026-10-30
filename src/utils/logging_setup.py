
import logging
import os
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    os.makedirs("data", exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data/app.log", encoding="utf-8")
        ]
    )
    return logging.getLogger("SupportPearlz")

logger = setup_logging()