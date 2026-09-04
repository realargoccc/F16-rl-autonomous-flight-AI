import math 
import numpy as np
from .base import BaseReward, range_window

def ao_ta_range(me_pos, me_vel, foe_pos, foe_vel):
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

    def _orientation(self, ao, ta):
        return 1.0 / (50.0 * ao / np.pi + 2.0) + 0.5 \
                + min(np.arctanh(1.0 - max(2.0 * ta / np.pi, 1e-4)) / (2.0 * np.pi), 0.0) + 0.5

    def _range(self, range_km):
        return 1.0 * (range_km < 5) \
                + (range_km >=5) * np.clip(-0.032 * range_km ** 2 + 0.284 * range_km + 0.38, 0.0, 1.0) \
                + np.clip(np.exp(-0.16 * range_km), 0.0, 0.2)

    def raw(self, env, computed):
        ao, ta, range_m = ao_ta_range(env.me.pos(), env.me.vel(), env.foe.pos(), env.foe.vel())
        return self._orientation(ao, ta) * self._range(range_m / 1000.0)