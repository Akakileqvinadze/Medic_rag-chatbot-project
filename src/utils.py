import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error running {func.__name__}: {e}")
        return None
