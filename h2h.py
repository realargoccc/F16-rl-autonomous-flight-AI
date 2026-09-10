import numpy as np
from flight_env import F16Env
from selector import Selector

class _Raw():
    mean, var = 0.0, 1.0 #normalizer

def as_foe(select):
    return (select, _Raw, np.inf, 0.0)  #(model, rms, clip, eps)

def make_h2h(aspect_band=(0.0, 80.0), range_band=(700.0, 1200.0)):
    env = F16Env()
    env.aspect_band = aspect_band
    env.range_band = range_band
    return env, Selector(), Selector()

def fight(env, me, foe, seed):
    obs, _ = env.reset(seed=seed)
    env.foe_policy = as_foe(foe)
    me.reset()
    foe.reset()
    while True:
        a, _ = me.predict(obs)
        obs, r, terminate, truncate, info = env.step(a)
        if terminate or truncate:
            break
    end = ("win" if info["win"] else "killed" if env.agent_hp <= 0
           else "crash" if info["crashed"] else "deck" if info["deck_hit"]
           else "foe_down" if info["foe_down"] else "draw")
    return end, me.switches, foe.switches
        