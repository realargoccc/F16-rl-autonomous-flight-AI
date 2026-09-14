'''selector training env, decision pools includes offense (v4.1.6), defense(v1.0.2), h2h(v1.0.1)'''

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from flight_env import F16Env
from selector import Selector 