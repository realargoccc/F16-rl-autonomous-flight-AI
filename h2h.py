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