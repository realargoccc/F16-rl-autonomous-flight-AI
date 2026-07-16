import jsbsim
import os
from flight_env import F16Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

model_load = "ppo_f16_eleva_v2.0.6.zip"         #COMMEWNT OUT WHEN TRAIN FRESH, UN COMMENT WHEN TRAIN CONTINUOUS
model_path = "ppo_f16_eleva_v2.1.0.zip" 
vecnorm_load = "vecnorm_eleva_v2.0.6.pkl"       #COMMEWNT OUT WHEN TRAIN FRESH, UN COMMENT WHEN TRAIN CONTINUOUS
vecnorm_path = "vecnorm_eleva_v2.1.0.pkl"

#sanity check 
#env = F16Env()
if __name__ == "__main__":
    check_env(F16Env())
    env = SubprocVecEnv([lambda: Monitor(F16Env()) for _ in range(8)])   #auto wrap 
    
    env = VecNormalize(         #COMMEWNT OUT WHEN TRAIN CONTINUOUS, UNCOMMENT WHEN TRAIN FRESH
        env, 
        norm_obs=True,          #normalize observations
        norm_reward=False,      #DO NOT normalize reward since they are specifically assigned
        clip_obs=10.0           #cap the upper and lower limit between -10 - 10
    )
    
    #env = VecNormalize.load(vecnorm_load, env)  #COMMEWNT OUT WHEN TRAIN FRESH, UN COMMENT WHEN TRAIN CONTINUOUS
    #env.training = True                         #COMMEWNT OUT WHEN TRAIN FRESH, UN COMMENT WHEN TRAIN CONTINUOUS
    #env.norm_reward = False                     #COMMEWNT OUT WHEN TRAIN FRESH, UN COMMENT WHEN TRAIN CONTINUOUS

    #tensorboard --logdir=./tb_logs/
    #model = PPO.load(model_load, env=env, ent_coef = 0.02, verbose = 1, tensorboard_log="./tb_logs/")
    model = PPO("MlpPolicy", env, verbose = 1, n_steps=512, batch_size=1024, gamma = 0.99, ent_coef = 0.01, tensorboard_log="./tb_logs/") #ent_coef controls how much PPO encourage exploration 

    model.learn(total_timesteps= 1_600_000, tb_log_name="v2.0.8") #reset_num_timesteps=False (Add when train continous, remove when train fresh)
    model.save(model_path)
    env.save(vecnorm_path)


#Building environment: F16Env -> DummyVecEnv -> VecNormalize
#env = DummyVecEnv([lambda: F16Env()])

#Wrapping existed one or starting a new one:
#if os.path.exists(vecnorm_path):
    #env = VecNormalize.load(vecnorm_path, env)
#else:     #norm obs balanced all values, so no values stand out, but don't normalize reward
    #env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)    

#pull up the existed ppo path, if doesn't exist, create a new one 

#interpretor select command: /Users/y/Desktop/jsbsim-rl/.venv/bin/python
