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