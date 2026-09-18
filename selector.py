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
        frozen snap version: v1.1.1
    '''

    BORESIGHT, FOE_BS = 23, 29   #obs location, to switch
    RANGE, CLOSURE = 18, 21
    MODES = ("offense", "defense", "merge")

    def __init__(self, off_tag="eleva_v4.1.6", def_tag="def_v1.0.2", merge_tag="eleva_h2h_v1.0.1", 
                 snap_tag=None, snap_range = 2500.0, snap_closure=300.0, min_dwell=10, offense_dwell=50):
        self.experts = {"offense": load_policy(off_tag),
                        "defense": load_policy(def_tag),
                        "merge": load_policy(merge_tag)
                        }
        if snap_tag is not None:
            self.experts["snap"] = load_policy(snap_tag)
        self.snap_range, self.snap_closure = snap_range, snap_closure
        self.offense_dwell=offense_dwell
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
        if "snap" in self.experts and obs[self.RANGE] < self.snap_range and obs[self.CLOSURE] > self.snap_closure:
            return "snap"   
        return "merge"
    
    def _pick(self, obs):      
        choice = self.quadrant(obs)
        need = self.offense_dwell if self.mode == "offense" else self.min_dwell
        if self.mode is None:      #first decision
            self.mode = choice
        elif choice != self.mode and (choice == "defense" or self.held >= need):
            self.mode = choice
            self.held = 0
            self.switches += 1
        self.held += 1
        return self.mode

    def act (self, mode, obs, deterministic=True):
        '''fly with one expert at a time with the name provided'''
        model, rms, clip, eps = self.experts[mode]
        n = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip)
        a, _ = model.predict(n.astype(np.float32), deterministic=deterministic)
        return a

    def predict(self, obs, deterministic=True):
        return self.act(self._pick(obs), obs, deterministic), None