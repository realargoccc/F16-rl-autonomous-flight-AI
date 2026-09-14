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

wins = crashes = earned = given = 0
best = None
for seed in SEEDS:
    obs, _ = raw.reset(seed=seed)
    quads, rows = [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        obs, reward, terminated, truncated, info = raw.step(action)
        quads.append(quadrant(obs[23], obs[29]))
        me, foe = raw.me.pos(), raw.foe.pos()
        rows.append({                       #only what csvtotacview reads
            "time": raw.me.get_sim_time(),
            "bank_rad": raw.me['attitude/phi-rad'],
            "pitch_rad": raw.me['attitude/theta-deg'],   #degrees, csvtotacview passes it through
            "heading_deg": raw.me['attitude/psi-deg'],
            "agent_n_m": float(me[0]), "agent_e_m": float(me[1]), "agent_up_m": float(me[2]),
            "foe_n_m": float(foe[0]), "foe_e_m": float(foe[1]), "foe_up_m": float(foe[2]),
            "foe_roll_deg": raw.foe['attitude/phi-deg'],
            "foe_pitch_deg": raw.foe['attitude/theta-deg'],
            "foe_yaw_deg": raw.foe['attitude/psi-deg'],
        })

        