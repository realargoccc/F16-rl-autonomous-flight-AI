from flight_env import F16Env
from stable_baselines3.common.monitor import Monitor

ASPECT_BAND = (150.0, 180.0)
RANGE_BAND = (2500.0, 4000.0)

def make_neutral_env(aspect_band=ASPECT_BAND, range_band=RANGE_BAND):
    env = F16Env()
    env.aspect_band = aspect_band
    env.range_band = range_band
    env.defensive_p = 0.0
    env.foe_pool_prob = 0.0

    return Monitor(env, info_keywords=("crashed", "foe_crashed", "win", "deck_hit", "foe_down"))

