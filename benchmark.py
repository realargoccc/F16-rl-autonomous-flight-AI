'''
    Objective: one fixed way to measure a checkpoint, so two numbers can actually be compared.

    Purpose: every time the env changes something silently the win rate moves and I dont find out
    for days. This runs the same 180 fights every time on the same seeds, so if the number moves
    it is either the policy or the env, not the dice.

    How: two seed blocks (9000 and 20000), report per cell not aggregate, paired McNemar when
    comparing two checkpoints.

    Rule: dont change the seeds or the cell definition. If they change all the old numbers are dead.
'''
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from flight_env import F16Env 

model_version = "v2.7.5"
compare_vers = None
baseline = 0.62         #last seen good version
tol = 0.05              #how far it can drift

seeds = list(range(9000, 9090)) + list(range(20000, 20090))
aspect_edges = [20.0, 40.0, 60.0, 80.0]
aspect_names = ["0-20", "20-40", "40-60", "60-80"]
tactics = ["flee", "beam+", "beam-"]
pitches = ["climb", "level", "dive"]

def run_sweep(tag):
    model = PPO.load("ppo_f16_eleva_" + tag + ".zip")
    tmp = DummyVecEnv([lambda: F16Env()])
    vecnorm = VecNormalize.load("vecnorm_eleva_" + tag + ".pkl", tmp)
    vecnorm.training = False
    vecnorm.norm_reward = False
    raw = F16Env()
    rows = []

    for s in seeds:
        obs, info = raw.reset(seed=s)
        if raw.turn_offset > 2.0: tac = "flee"
        elif raw.turn_offset > 0: tac = "beam+"
        else: tac = "beam-"

        if raw.pitch_target > 0.01: pit = "climb"
        elif raw.pitch_target < -0.01: pit = "dive"
        else: pit = "level"

        aa = abs(raw.spawn_aspect) #aspect angle
        asp = aspect_names[-1]
        for i in range(len(aspect_edges)):
            if aa < aspect_edges[i]:
                asp = aspect_names[i]
                break

        r0 = raw.range      #range at spawn
        rmin = r0
        dwell = 0           #steps in wez(range)
        bs_list = []        #boresight steps in wez(range)
        crashed = False
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, info = raw.step(action)

            if raw.range < rmin:
                rmin = raw.range
            if raw.gun_rmin <= raw.range <= raw.gun_rmax:
                dwell += 1
                bs_list.append(np.degrees(raw.boresight))
            if raw.me['position/h-agl-ft'] * 0.3048 < 30 or abs(raw.me['accelerations/Nz']) > 13.0:
                crashed = True
        if len(bs_list) > 0:    #if never enter wez, need a bad run (warning)
            medbs = float(np.median(bs_list))
        else:
            medbs = np.nan

        rows.append({"seed": s, "tac": tac, "pit": pit, "asp":asp,
                        "win": bool(raw.foe_hp <= 0.0), 
                        "crashed": crashed, 
                        "spawn_range": r0, 
                        "min_range": rmin,
                        "dwell": dwell,
                        "foe_hp": float(raw.foe_hp),
                        "medbs": medbs,})
    return rows

#helpers
def count_cell(rows, key, value):
    w = 0
    n = 0
    for r in rows:
        if r[key] == value:
            n += 1
            if r["win"]:
                w += 1
    return w, n

def count_pair(rows, tac, pit):
    w = 0
    n = 0
    for r in rows:
        if r["tac"] == tac and r["pit"] == pit:
            n += 1
            if r["win"]:
                w += 1
    return w, n

