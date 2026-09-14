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
    '''frozen offense version: v4.1.56
        frozen defense version: v1.0.2
        frozen merge version: v1.0.1
    '''

    BORESIGHT, FOE_BS = 23, 29   #obs location, to switch

    def __init__(self, off_tag="eleva_v4.1.6", def_tag="def_v1.0.2", merge_tag="eleva_h2h_v1.0.1", min_dwell=10):
        self.experts = {"offense": load_policy(off_tag),
                        "defense": load_policy(def_tag),
                        "merge": load_policy(merge_tag)}
        self.min_dwell = min_dwell  #decide every 1s
        self.reset()

    def reset(self):
        self.mode = None
        self.held = 0
        self.switches = 0

    def quadrant(self, obs):
        half = np.pi / 2
        agent, foe = obs[self.BORESIGHT], obs[self.FOE_BS]
        if agent < half and foe > half: return "offense"
        if agent > half and foe < half: return "defense"
        return "merge"
    
    def _pick(self, obs):      
        choice = self.quadrant(obs)
        if self.mode is None:      #first decision
            self.mode = choice
        elif choice != self.mode and (choice == "defense" or self.held >= self.min_dwell):
            self.mode = choice
            self.held = 0
            self.switches += 1
        self.held += 1
        return self.mode

    def predict(self, obs):
        mode = self._pick(obs)
        model, rms, clip, eps = self.off if mode == "offense" else self.dfn
        n = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip)
        a, _ = model.predict(n.astype(np.float32), deterministic=True)
        return a, None