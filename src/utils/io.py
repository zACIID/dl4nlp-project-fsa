from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent

SRC_DIR = PROJECT_ROOT / 'src'

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATASET_DIR = DATA_DIR / 'raw'


# Reference: https://stackoverflow.com/a/53877507
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
