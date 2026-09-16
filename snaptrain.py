from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from pisnap import make_snap_env

model_load = "ppo_f16_eleva_v4.1.6.zip"      
vecnorm_load = "vecnorm_eleva_v4.1.6.pkl"    
TAG = "snap_v1.0.0"
model_path = "ppo_f16_" + TAG + ".zip"
vecnorm_path = "vecnorm_" + TAG + ".pkl"

TAKEN_WEIGHT = None
total_steps = 1_000_000

if __name__ == "__main__":
    env = SubprocVecEnv([partial(make_snap_env, opponent="pursuit", taken_weight=TAKEN_WEIGHT) for _ in range(8)])
    env = VecNormalize.load(vecnorm_load, env)
    env.training = True
    env.norm_reward = False

    model = PPO.load(model_load, env=env, ent_coef=1e-3, verbose=1, tensorboard_log="./tb_logs/")
    model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name=TAG)
    model.save(model_path)
    env.save(vecnorm_path)