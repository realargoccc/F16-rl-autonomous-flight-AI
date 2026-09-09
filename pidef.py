#defensive environment — everything inherited from F16Env except _reward
#obs / physics / actions / termination must stay identical or the selector can't hand over mid-episode

import math
import numpy as np
from flight_env import F16Env
from stable_baselines3.common.monitor import Monitor

FOE_POOL = ["v4.1.5"]

ASPECT_BAND = (60.0, 90.0)
RANGE_BAND = (2500.0, 4000.0)

def make_def_env(attackers=FOE_POOL, aspect_band=ASPECT_BAND, range_band=RANGE_BAND):
    env = F16Env()
    env.defensive_p = 1.0
    env.foe_pool_prob = 1.0
    env.aspect_band = aspect_band
    env.range_band = range_band
    for tag in attackers:
        env.load_foe(tag)
    return Monitor(env, info_keywords = ("crashed", "foe_crashed", "win", "deck_hit", "foe_down"))