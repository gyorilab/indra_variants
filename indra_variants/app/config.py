import os
from pathlib import Path

HERE = Path(__file__).parent.resolve()
configured_data_dir = Path(os.getenv("DATA_DIR") or HERE.parent.joinpath("data"))
DATA_DIR = configured_data_dir.joinpath("variant_effect/variant_effect")
PORT = int(os.getenv("PORT") or os.getenv("DASH_PORT", 8051))
DEBUG = bool(int(os.getenv("DASH_DEBUG", 0)))
