from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from pih2h import make_h2h_env
from train import SelfPlayPool

model_load = "ppo_f16_eleva_v4.1.6.zip"
model_path = "ppo_f16_eleva_h2h_v1.0.0.zip"
vecnorm_load = "vecnorm_eleva_v4.1.6.pkl"
vecnorm_path = "vecnorm_eleva_h2h_v1.0.0.pkl"

snapshot_every = 500_000    #500k per snapshot (update selfplay version)
snapshot_warm = 0        
pool_prob = 1.0  

if __name__ == "__main__":
    env = SubprocVecEnv([make_h2h_env for _ in range(8)])
    env = VecNormalize.load(vecnorm_load, env)
    env.training = True
    env.norm_reward = False

    model = PPO.load(model_load, env=env, ent_coef=1e-3, verbose=1, tensorboard_log="./tb_logs/")

    pool = SelfPlayPool(every=snapshot_every, prefix="h2h_v1.0.0", warmup=snapshot_warm, prob=pool_prob)
    model.learn(total_timesteps=3_000_000, callback=pool, reset_num_timesteps=False, tb_log_name="h2h_v1.0.0")
    model.save(model_path)
    env.save(vecnorm_path)