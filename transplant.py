import pickle, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from flight_env import F16Env
import os, time, torch

old_model = "ppo_f16_eleva_v2.3.0.zip"
old_vec = "vecnorm_eleva_v2.3.0.pkl"
new_model = "ppo_f16_eleva_v2.3.1.zip"
new_vec = "vecnorm_eleva_v2.3.1.pkl"

old_dim = 24, new_dim = 40

assert os.path.exist(old_model), f"missing {old_model}"
assert os.path.exist(old_vec),   f"missing {old_vec}"

old_poli = PPO.load(old_model, device = "cpu").policy.state_dict() #old trained policy 

#fresh policy with new_dim
env = VecNormalize(DummyVecEnv([lambda: F16Env()]), norm_obs=True, norm_reward=False, clip_obs=10.0)

assert env.observation_space.shape == (new_dim, ), f"obs dim doesn't match, should be {new_dim}"

new = PPO("MlpPolicy", env, n_steps=512, batch_size=1024, gamma = 0.997, ent_coef = 0.01, verbose=0)

new_poli = new.policy.state_dict()

#surgery 
copied = widened = skipped = 0
for k in new_poli: 
    if k not in old_poli:
        skipped += 1
        continue
    if new_poli[k].shape == old_poli[k].shape:
        new_poli[k] = old_poli[k].clone()
        copied += 1
    elif (new_poli[k].dim() == 2 and new_poli[k].shape[1] == new_dim and old_poli[k].shape[1] == old_dim):
        w = torch.zeros_like(new_poli[k])
        w[:, :old_dim] = old_poli[k]
        new_poli[k] = w
        widened += 1
    else:
        raise RuntimeError(f"wrong shape for {k}: " 
                           f"old {tuple(old_poli[k].shape)} vs new {tuple(new_poli[k].shape)}")
assert widened >= 2, f"expected >= 2 widened layers, but have {widened}"
new.policy.load_state_dict(new_poli)
new.save(new_model)

#extend vecnormalize stats
with open(old_vec, "rb") as f:
    old_vn = pickle.load(f)

pad = new_dim - old_dim
env.obs_rms.mean = np.concatenate([old_vn.obsrms.mean, np.zeros(pad)])
env.obs_rms.var = np.concatenate([old_vn.obsrms.var, np.ones(pad)])
env.obs_rms.count = old_vn.obs_rms.count
env.save(new_vec)