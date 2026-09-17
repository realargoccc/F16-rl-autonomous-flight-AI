from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from pisnap import make_snap_env

model_load = "ppo_f16_snap_v1.1.0.zip"      
vecnorm_load = "vecnorm_snap_v1.1.0.pkl"    
TAG = "snap_v1.1.1"
model_path = "ppo_f16_" + TAG + ".zip"
vecnorm_path = "vecnorm_" + TAG + ".pkl"

TAKEN_WEIGHT = None
REL_ALT_BAND = (-500.0, 500.0)
total_steps = 1_000_000

if __name__ == "__main__":
    env = SubprocVecEnv([partial(make_snap_env, opponent="pursuit", taken_weight=TAKEN_WEIGHT, 
                                 rel_alt_band=REL_ALT_BAND) for _ in range(8)])
    
    env = VecNormalize.load(vecnorm_load, env)
    env.training = True
    env.norm_reward = False
    model = PPO.load(model_load, env=env, ent_coef=1e-3, verbose=1, tensorboard_log="./tb_logs/")
    '''
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=1024, n_epochs=10,
              gamma=0.997, gae_lambda=0.95, clip_range=0.2, ent_coef=1e-3, vf_coef=0.5,
              max_grad_norm=0.5, verbose=1, tensorboard_log="./tb_logs/")
    '''          
    model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name=TAG) #reset_num_timesteps=False,
    model.save(model_path)
    env.save(vecnorm_path)