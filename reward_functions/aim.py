import math
from base import range_window, BaseReward

class Aim(BaseReward):
    ''' gun cone is 3deg, continuous aim is flat'''

    def _value(self, env, bs): #bs = boresight
        excess = max(0.0, bs - env.gun_cone)
        approach = math.exp(-excess / env.aim_width)
        in_cone = max(0.0, 1.0 - bs / env.gun_cone)
        return approach + env.k_cone * in_cone

    def raw(self, env, computed):
        w = range_window(env, env.range)
        return env.k_aim * w * (self._value(env, env.boresight) - self._value(env, computed.foe_boresight))