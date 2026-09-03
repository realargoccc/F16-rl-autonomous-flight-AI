import math 
import numpy as np
from base import BaseReward

class Posture(BaseReward):
    '''orientation * range, as a potential'''
    is_potential = True

    def __init__(self, env):
        super().__init__(env)
        self.scale = env.k_bridge

    def _orientation(self, env, computed):
        eta_ata = 1.0 - env.boresight / np.pi           #1.0 = nose on nose
        eta_aa  = env.aspect_angle / np.pi              #1.0 = nose on tail
        agent_adv = 0.5 * eta_ata + 0.5 * eta_aa

        foe_eta_ata = 1.0 - computed.foe_boresight / np.pi
        foe_eta_aa  = env.foe_state.aspect_angle / np.pi
        foe_adv = 0.5 * foe_eta_ata + 0.5 * foe_eta_aa

        return agent_adv - foe_adv                      #[-1, 1]

    def _range(self, env):
        range = env.range
        #out of band
        approach = math.exp(-max(0.0, range - env.gun_rmax) / env.range_width)
        #in the band
        if range < env.gun_rmin:
            in_band = range / env.gun_rmin
        else:
            in_band = max(0.0, (env.gun_rmax - range) / (env.gun_rmax - env.gun_rmin))

        return (approach + env.k_close * in_band) / (1.0 + env.k_close)

    def raw(self, env, computed):
        return self._orientation(env, computed) * self._range(env)