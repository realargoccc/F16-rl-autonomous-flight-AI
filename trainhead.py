import torch
from heading_env import HeadingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

model_load = "controller_v1.0.0.zip"
model_path = "controller_v1.0.1.zip"

def make_env():
    return Monitor(HeadingEnv(), info_keywords=("crashed", "deck_hit", "missed", "turn_counts"))

if __name__ == "__main__":
    check_env(HeadingEnv())
    env = SubprocVecEnv([make_env for _ in range(8)])   #no vecnorm since obs normalises itself

    #model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=1024,
    #            gamma=0.995, ent_coef=1e-3, tensorboard_log="./tb_logs/")
    
    model = PPO.load(model_load, env=env, ent_coef = 1e-3, verbose=1, tensorboard_log="./tb_logs/")

    model.learn(total_timesteps=5_000_000, tb_log_name="controller_v1.0.0")
    model.save(model_path)