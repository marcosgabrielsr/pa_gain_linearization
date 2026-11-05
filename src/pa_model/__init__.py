from pathlib import Path
from .dataset import Dataset
from .paths import ROOT, DATA_RAW, DATA_PROCESSED, RESULTS, CHECKPOINTS, LOGS
from .sd_window import SWDataset, WindowLoader

Path(RESULTS).mkdir(exist_ok=True)