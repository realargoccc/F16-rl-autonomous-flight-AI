import jsbsim
import os, torch
from flight_env import F16Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

model_load = "ppo_f16_eleva_v4.0.0.zip"         #COMMEWNT OUT WHEN TRAIN FRESH, UN COMMENT WHEN TRAIN CONTINUOUS
model_path = "ppo_f16_eleva_v4.0.5.zip" 
vecnorm_load = "vecnorm_eleva_v4.0.0.pkl"       #COMMEWNT OUT WHEN TRAIN FRESH, UN COMMENT WHEN TRAIN CONTINUOUS
vecnorm_path = "vecnorm_eleva_v4.0.5.pkl"

#selfplay 
snapshot_every = 500_000
snapshot_warm = 0
pool_prob = 0.5

class SelfPlayPool(BaseCallback):
    def __init__(self, every, prefix, warmup, prob=0.5, verbose=1):
        super().__init__(verbose)
        self.every, self.prefix, self.warmup, self.prob= every, prefix, warmup, prob
        self.gen = 0

    def _on_training_start(self):
        self.next_at = self.num_timesteps + self.warmup

    def _on_step(self):
        if self.num_timesteps < self.next_at:
            return True
        self.next_at += self.every
        tag = f"{self.prefix}_g{self.gen:02d}"
        self.model.save("ppo_f16_eleva_" + tag + ".zip")
        self.training_env.save("vecnorm_eleva_" + tag + ".pkl")
        self.training_env.env_method("load_foe", tag)
        self.training_env.set_attr("foe_pool_prob", self.prob)
        self.gen += 1
        if self.verbose:
            print(f"[selfplay] gen {self.gen} snapshot {tag} at {self.num_timesteps} steps")
        return True
    
def make_env():
    env = F16Env()
    env.aspect_band = (0.0, 80.0)
    env.defensive_p = 0.0
    #foe_pool = ["v2.8.0", "v2.8.1", "v2.8.2", "v2.8.3", "v2.8.4", "v2.8.5", "v2.8.6", "v2.8.7", "v2.8.8"]
    #for vers in foe_pool:
    #    env.load_foe(vers)
    env.foe_pool_prob = 0.0
    env.selfplay = "pfsp"
    return Monitor(env, info_keywords=("crashed", "foe_crashed", "win", "deck_hit", "foe_down"))

if __name__ == "__main__":
    check_env(F16Env())
    env = SubprocVecEnv([make_env for _ in range(8)])   #auto wrap 
    
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
    #model = PPO.load(model_load, env=env, ent_coef = 0.002, verbose = 1, tensorboard_log="./tb_logs/")
    model = PPO("MlpPolicy", env, verbose = 1, n_steps=512, batch_size=1024, gamma = 0.997, ent_coef = 1e-3, tensorboard_log="./tb_logs/") #ent_coef controls how much PPO encourage exploration 

    pool = SelfPlayPool(every=snapshot_every, prefix="v4.0.3", warmup=snapshot_warm, prob=pool_prob)
    model.learn(total_timesteps= 3_000_000, tb_log_name="v4.0.3") #reset_num_timesteps=False (Add when train continous, remove when train fresh)
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
