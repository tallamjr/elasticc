import os
import platform
from pathlib import Path

try:
    ROOT = f"{os.environ['ROOT']}"
except Exception as e:
    print(f"Environment variable not set:{e}.\nDefining relative to constants.py file")
    ROOT = Path(__file__).absolute().parent.parent

SYSTEM = platform.system()
# Linux: Linux
# Mac: Darwin
# Windows: Windows
