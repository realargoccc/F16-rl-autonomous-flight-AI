import torch
from pidef import make_def_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

model_load   = "ppo_f16_eleva_v4.1.5.zip"     
model_path   = "ppo_f16_def_v1.0.0.zip"   
vecnorm_load = "vecnorm_eleva_v4.1.5.pkl"
vecnorm_path = "vecnorm_def_v1.0.0.pkl"

if __name__ == "__main__":
    check_env(make_def_env().env)
    env = SubprocVecEnv([make_def_env for _ in range(8)])
    env = VecNormalize.load(vecnorm_load, env)
    env.training = True
    env.norm_reward = False
    #env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    #model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=1024, gamma=0.997, ent_coef=0.03, tensorboard_log="./tb_logs/")
    model = PPO.load(model_load, env=env, ent_coef=1e-3, verbose=1, tensorboard_log="./tb_logs/")
    
    model.learn(total_timesteps=3_000_000,reset_num_timesteps=False, tb_log_name="def_v1.0.0") #reset_num_timesteps=False,
    model.save(model_path)
    env.save(vecnorm_path)