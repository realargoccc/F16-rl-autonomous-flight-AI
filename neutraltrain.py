from pineutral import make_neutral_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize 

model_load = "ppo_f16_eleva_v4.1.6.zip"      #warm from best offense train
vecnorm_load = "vecnorm_eleva_v4.1.6.pkl"    
model_path = "ppo_f16_neutral_v1.0.0.zip"    
vecnorm_path = "vecnorm_neutral_v1.0.0.pkl" 

if __name__ == "__main__":
    env = SubprocVecEnv([make_neutral_env for _ in range(8)])
    env = VecNormalize.load(vecnorm_load, env)
    env.training = True
    env.norm_reward = False

    model = PPO.load(model_load, env=env, ent_coef = 1e-3, verbose=1, tensorboard_log="./tb_logs/") #no selfplay
    model.learn(total_timesteps=3_000_000, reset_num_timesteps=False, tb_log_name="neutral_v1.0.0")
    model.save(model_path)
    env.save(vecnorm_path)
