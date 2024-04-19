import psutil
import multiprocessing as mp

MAX_AVAILABLE_CORES: int = mp.cpu_count()
MAX_AVAILABLE_RAM_GB: int = psutil.virtual_memory().total // 10 ** 9
