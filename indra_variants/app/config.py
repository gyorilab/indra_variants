import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "variant_effect"))
PORT     = int(os.getenv("DASH_PORT", 8051))
DEBUG    = bool(int(os.getenv("DASH_DEBUG", 0)))