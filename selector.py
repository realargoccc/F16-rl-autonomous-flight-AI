'''switch between offense and defense during the fight'''

import pickle
import numpy as np
from stable_baselines3 import PPO

def load_policy(tag):
    '''full name of the model version'''
    model = PPO.load("ppo_f16_" + tag + ".zip", device="cpu")
    with open("vecnorm_" + tag + ".pkl", "rb") as fh:
        vn = pickle.load(fh)
    return model, vn.obs_rms, float(vn.clip_obs), float(vn.epsilon)

class Selector:
    '''"frozen offense version: v4.1.5
        frozen defense version: v1.0.2
    '''

    RANGE, BORESIGHT, FOE_BS = 18, 23, 29   #obs location, to switch

    def __init__(self, off_tag="eleva_4.1.5", def_tag="def_v1.0.2", threat_range=2500.0, margin=np.radians(20.0)):
        self.off = load_policy(off_tag)
        self.dfn = load_policy(def_tag)
        self.threat_range = threat_range
        self.margin = margin

    def reset(self):
        self.mode = "offense"
        self.switches = 0

    def _pick(self, obs):
        rng, own, bandit = obs[self.RANGE], obs[self.BORESIGHT], obs[self.FOE_BS]
        if rng > self.threat_range:      #too far
            new_po = "offense"
        elif bandit < own - self.margin: #bandit pointing at agent
            new_po = "defense"
        elif own < bandit - self.margin:
            new_po = "offense"           #agent pointing at bandit 
        else:
            new_po = self.mode
        if new_po != self.mode:
            self.swithces += 1
            self.mode = new_po
        return self.mode