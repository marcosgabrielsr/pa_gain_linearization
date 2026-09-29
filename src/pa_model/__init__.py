from pathlib import Path
from .dataset import Dataset
from .paths import ROOT, DATA_RAW, DATA_PROCESSED, RESULTS, CHECKPOINTS, LOGS
from .sliding_window import SlidingWindowDataset

Path(RESULTS).mkdir(exist_ok=True)