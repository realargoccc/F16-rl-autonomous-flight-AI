import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from flight_env import F16Env

class SnapEnv(gym.Env):
    def __init__(self, aspect_band = (175.0, 180.0), range_band = (2500.0, 4000.0), bearing_spread=5.0, opponent="pursuit", max_steps=300
                 ,taken_weight=None):
        self.env = F16Env()
        self.env.aspect_band = aspect_band          
        self.env.range_band = range_band
        self.env.bearing_spread = bearing_spread   
        self.env.defensive_p = 0.0
        self.env.max_episodes_steps = max_steps
        self.opponent = opponent
        if opponent == "v4.1.6":
            self.env.load_foe("v4.1.6")
        self.taken_weight = self.env.k_damage if taken_weight is None else taken_weight
        self.env.foe_pool_prob = 1.0 if opponent == "v4.1.6" else 0.0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        if self.opponent == "pursuit":
            self.env.turn_offset = 0.0  #bandit steers at the agent rather than flee
            self.env.pitch_target = 0.0
        self.inside = False
        return obs, info

    def step(self, action):
        hp_before = self.env.agent_hp
        obs, reward, terminated, truncated, info = self.env.step(action)
        taken = hp_before - self.env.agent_hp
        reward += (self.env.k_damage - self.taken_weight) * taken
        if self.env.range <= self.env.gun_rmax:
            self.inside = True
        info["pass_done"] = self.inside and self.env.range > self.env.gun_rmax
        return obs, reward, terminated or info["pass_done"], truncated, info

def make_snap_env(**kwargs):
    return Monitor(SnapEnv(**kwargs), info_keywords=("crashed", "foe_crashed", "win", "deck_hit", "foe_down", "pass_done"))