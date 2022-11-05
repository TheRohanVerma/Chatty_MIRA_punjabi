import regex 
import random

#should be added
import torch
import torch.nn.functional as F
import ktrain
from ktrain import text
import pandas as pd

class ActionEliza():
    # a =  5
    def __init__(self):
        self.a = 'ਚੰਗੇ ਬੋਟ'

    def sum(self, b):
        if self.a in b:
            return 'found something'
        else:
            return 'nopsie'





