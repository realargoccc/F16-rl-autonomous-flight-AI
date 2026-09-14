from flight_env import F16Env
from piselect import make_select_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

TAG = "select_v1.0.0"
model_path = "ppo_f16_" + TAG + ".zip"
vecnorm_path = "vecnorm_" + TAG + ".pkl"

total_decisions = 300_000
snapshot = 50_000

class SnapshotPool(BaseCallback):
    def __init__(self, every, prefix, verbose=1):
        super().__init__(verbose)
        self.every, self.prefix = every, prefix
        self.gen = 0

    def _on_training_start(self):
        self.next_at = self.num_timesteps + self.every

    def _on_step(self):
        if self.num_timesteps < self.next_at:
            return True
        self.next_at += self.every
        tag = f"{self.prefix}_g{self.gen:02d}"
        self.model.save("ppo_f16_" + tag + ".zip")
        self.training_env.save("vecnorm_" + tag + ".pkl")
        self.training_env.env_method("add_learned_opponent", tag)
        self.gen += 1
        if self.verbose:
            print(f"[snapshot] {tag} at {self.num_timesteps} decisions, added to every worker's pool")

        return True

if __name__ == "__main__":
    env = SubprocVecEnv([make_select_env for _ in range(8)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO("MlpPolicy", env, gamma=0.97, n_steps=256, batch_size=512, ent_coef=0.01, verbose=1, tensorboard_log="./tb_logs/")
    pool = SnapshotPool(every=snapshot, prefix=TAG)
    model.learn(total_timesteps=total_decisions, callback=pool, tb_log_name=TAG)
    model.save(model_path)
    env.save(vecnorm_path)