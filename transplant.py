import pickle, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from flight_env import F16Env
import os, time, torch

"""
Grow observations space without retraining fresh

WHY: Adding more observations cut off the oppourtunity to train warm from a good behavioring agent, causing delay and waste of computaion worth
of hours

HOW: only 2 tensors rely on obs size(policy net + value net first layers)
     copy every other tensor as is, for the 2 tensors, past ethe trained weights into the left columns and leave the new columns at zeros
     zero weights -> new inputs contribute nothing -> the new policy computes the exact same function as the old one. 
     No performance loss, then training grows the weights off zero only if the new inputs help.

Only run once per obs dim size change

old_model must be the latest trained model 
"""

old_model = "ppo_f16_eleva_v2.7.6.zip"
old_vec = "vecnorm_eleva_v2.7.6.pkl"
new_model = "ppo_f16_eleva_v2.8.0.zip"
new_vec = "vecnorm_eleva_v2.8.0.pkl"

old_dim, new_dim = 29, 30

assert os.path.exists(old_model), f"missing {old_model}"
assert os.path.exists(old_vec),   f"missing {old_vec}"

old_ppo = PPO.load(old_model, device="cpu")
old_poli = old_ppo.policy.state_dict() 

#fresh policy with new_dim
env = VecNormalize(DummyVecEnv([lambda: F16Env()]), norm_obs=True, norm_reward=False, clip_obs=10.0)

assert env.observation_space.shape == (new_dim, ), f"obs dim doesn't match, should be {new_dim}"

new = PPO("MlpPolicy", env, n_steps=512, batch_size=1024, gamma = 0.997, ent_coef = 0.01, verbose=0, device="cpu")

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
with torch.no_grad():
    new.policy.log_std.clamp_(max=-0.7)

new.num_timesteps = old_ppo.num_timesteps
new.save(new_model)

#extend vecnormalize stats
with open(old_vec, "rb") as f:
    old_vn = pickle.load(f)

pad = new_dim - old_dim
env.obs_rms.mean = np.concatenate([old_vn.obs_rms.mean, np.zeros(pad)])
env.obs_rms.var = np.concatenate([old_vn.obs_rms.var, np.ones(pad)])
env.obs_rms.count = old_vn.obs_rms.count
env.save(new_vec)

