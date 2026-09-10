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
        