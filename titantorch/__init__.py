__version__ = "2.0"


# TITAN optimizer
from .titan import AdamW as TITANtitusW
from .titan import AdamW as TITANtitusW8bit 

# GaLore optimizer
from .adafactor import Adafactor as GaLoreAdafactor
from .titusW import AdamW as GaLoreAdamW
from .titusW8bit import AdamW8bit as GaLoreAdamW8bit
