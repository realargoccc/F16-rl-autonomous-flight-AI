from pineutral import make_neutral_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize 

model_load = "ppo_f16_eleva_v4.1.6.zip"      #warm from best offense train
vecnorm_load = "vecnorm_eleva_v4.1.6.pkl"    
model_path = "ppo_f16_neutral_v1.0.0.zip"    
vecnorm_path = "vecnorm_neutral_v1.0.0.pkl" 