import math 
import numpy as np
from .base import BaseReward, range_window

def get_AO_TA_R(me_pos, me_vel, foe_pos, foe_vel):
    los = foe_pos - me_pos
    range_m = np.linalg.norm(los)
    me_speed = np.linalg.norm(me_vel)
    foe_speed = np.linalg.norm(foe_vel)

    ao = np.arccos(np.clip(np.dot(los, me_vel) / (range_m * me_speed + 1e-8), -1.0, 1.0))
    ta = np.arccos(np.clip(np.dot(los, foe_vel) / (range_m * foe_speed + 1e-8), -1.0, 1.0))

    return ao, ta, range_m

class Posture(BaseReward):
    '''orientation * range, as a potential'''
    is_potential = True

    def __init__(self, env):
        super().__init__(env)
        self.scale = env.k_bridge

    def _orientation(self, env, computed):
        

    def raw(self, env, computed):
        return self._orientation(env, computed) * range_window(env, env.range)