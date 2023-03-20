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

CLASS_MAPPING = {
    111: "SNIa",
    112: "SNIb/c",
    113: "SNII",
    114: "SNIax",
    115: "SN91bg",
    121: "KN",
    122: "M-dwarf Flare",
    123: "Dwarf Novae",
    124: "uLens",
    131: "SLSN",
    132: "TDE",
    133: "ILOT",
    134: "CART",
    135: "PISN",
    211: "Cepheid",
    212: "RR Lyrae",
    213: "Delta Scuti",
    214: "EB",
    215: "LPV/Mira",
    221: "AGN",
}
