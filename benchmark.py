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

    for s in seeds:
        obs, info = raw.reset(seed=s)
        if raw.turn_offset > 2.0: name = "flee"
        elif raw.turn_offset > 0: name = "beam+"
        else: tac = "beam-"

        if raw.pitch_target > 0.01: pit = "climb"
        elif raw.pitch_target < 0.01: pit = "dive"
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

        