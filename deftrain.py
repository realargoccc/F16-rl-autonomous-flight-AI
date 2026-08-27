import torch
from pidef import make_def_env, F16DefEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

#model_load   = "ppo_f16_eleva_v2.8.9.zip"     
model_path   = "ppo_f16_eleva_v3.0.0.zip"   
#vecnorm_load = "vecnorm_eleva_v2.8.9.pkl"
vecnorm_path = "vecnorm_eleva_v3.0.0.pkl"

def make_env():
    return Monitor(make_def_env(), info_keywords=("crashed", "foe_crashed", "win", "deck_hit"))

if __name__ == "__main__":
    check_env(F16DefEnv())
    env = SubprocVecEnv([make_env for _ in range(8)])
    #env = VecNormalize.load(vecnorm_load, env)
    #env.training = True
    #env.norm_reward = False
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=1024, gamma=0.997, ent_coef=0.03, tensorboard_log="./tb_logs/")

    '''
    model = PPO.load(model_load, env=env, ent_coef=0.002, verbose=1, tensorboard_log="./tb_logs/")
    with torch.no_grad():
        model.policy.log_std.fill_(-0.7)
    '''
    model.learn(total_timesteps=2_000_000, tb_log_name="def_v3.0.0") #reset_num_timesteps=False,
    model.save(model_path)
    env.save(vecnorm_path)