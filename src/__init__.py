import os
from .bp_guessing_decoder import bpgdg_decoder, bpgd_decoder, bp_history_decoder
from .osd_window import osd_window
from .bp4_osd import bp4_osd 
from . import __file__

def get_include():
    path = os.path.dirname(__file__)
    return path

f = open(get_include()+"/VERSION")
__version__ = f.read()
f.close()

