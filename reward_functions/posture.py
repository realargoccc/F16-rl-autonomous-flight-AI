import math 
import numpy as np
from base import BaseReward

class Posture(BaseReward):
    '''orientation * range, as a potential'''
    is_potential = True