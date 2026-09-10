from flight_env import F16Env
from stable_baselines3.common.monitor import Monitor

ASPECT_BAND = (150.0, 180.0) #merge only
RANGE_BAND = (2500.0, 4000.0)

def make_h2h_env(pool=(), aspect_band = ASPECT_BAND, range_band = RANGE_BAND):
    env = F16Env()
    env.aspect_band = aspect_band
    env.range_band = range_band
    env.defensive_p = 0.0
    env.selfplay = "pfsp"
    for tag in pool:
        env.load_foe(tag)
    env.foe_pool_prob = 1.0

    return Monitor(env, info_keywords=("crashed", "foe_crashed", "win", "deck_hit", "foe_down"))