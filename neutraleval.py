import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from flight_env import F16Env
from pineutral import make_neutral_env

TAG = "neutral_v1.0.0"
SEEDS = range(30000, 30060)       #merge ruler, never change
BASELINE = {"wins": 25, "earned_per_ep": 0.60, "earned_share": 0.51, "crashes": 4} 

def quadrant(bs, foe_bs):
    half = np.pi / 2
    if bs < half and foe_bs > half: return "OFF"
    if bs > half and foe_bs < half: return "DEF"
    if bs > half and foe_bs > half: return "SEP"
    return "MUT"

model = PPO.load("ppo_f16_" + TAG + ".zip")
vecnorm = VecNormalize.load("vecnorm_" + TAG + ".pkl", DummyVecEnv([lambda: F16Env()]))
vecnorm.training = False
vecnorm.norm_reward = False
raw = make_neutral_env().env

