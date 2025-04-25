import psutil
import torch
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def log_memory_usage():
    # Log system RAM memory
    ram = psutil.virtual_memory()
    logger.info(
        f"System RAM: {ram.used/1024**3:.0f}GB / "
        f"{ram.total/1024**3:.0f}GB / "
        f"{ram.available/1024**3:.0f}GB "
        f"(Used / Total / Free)"
    )

    # Log GPU memory if available
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        free = total - allocated
        stats = torch.cuda.memory_stats()
        fragmentation = stats["allocated_bytes.all.current"] / (
            stats["reserved_bytes.all.current"] + 1
        )
        logger.info(
            f"GPU Memory: {allocated/1024**2:.0f}MB / "
            f"{reserved/1024**2:.0f}MB / "
            f"{total/1024**2:.0f}MB / "
            f"{free/1024**2:.0f}MB "
            f"(Allocated / Cached / Total / Free) /"
            f"Fragmentation: {fragmentation:.2%}"
        )
