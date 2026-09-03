from collections import namedtuple
import math
StepComp = namedtuple("StepComp", ["speed_knots", "curr_g", "alt_agl_m", "crashed", "deck_hit", "truncated",
                                    "dmg_foe", "dmg_me", "foe_boresight"])

'''range halper for posture range'''

def range_window(env, range):
    if range < env.gun_rmin:
        return math.exp(-(env.gun_rmin - range) / env.close_width)
    if range > env.gun_rmax:
        return math.exp(-(range - env.gun_rmax) / env.range_width)
    return 1.0

class BaseReward:
    scale = 1.0
    is_potential = False

    def __init__(self, env):
        self.env = env
        self.prev = 0.0
        self.last = 0.0

    def reset(self, env, computed):
        self.prev = self.raw(env, computed) * self.scale if self.is_potential else 0.0
        self.last = 0.0

    def raw(self, env, computed):
        raise NotImplementedError

    def __call__(self, env, computed):
        r = self.raw(env, computed) * self.scale
        if self.is_potential:
            r, self.prev = r - self.prev, r
        self.last = r
        return r

    @property
    def name(self):
        return self.__class__.__name__