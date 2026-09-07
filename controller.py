import numpy as np
from stable_baselines3 import PPO

def control_obs(ac, tgt_alt, tgt_hdg, tgt_spd):
    alt        = ac['position/h-sl-meters']
    roll       = ac['attitude/phi-rad']
    pitch      = ac['attitude/theta-rad']
    heading    = ac['attitude/psi-rad']

    fwd_speed  = ac['velocities/u-fps'] * 0.3048
    side_speed = ac['velocities/v-fps'] * 0.3048
    vert_speed = ac['velocities/w-fps'] * 0.3048
    ias        = ac['velocities/vc-fps'] * 0.3048

    roll_rate  = ac['velocities/p-rad_sec']
    pitch_rate = ac['velocities/q-rad_sec']
    yaw_rate   = ac['velocities/r-rad_sec']

    d_alt = tgt_alt - alt
    d_hdg = (tgt_hdg - heading + np.pi) % (2 * np.pi) - np.pi
    d_spd = tgt_spd - fwd_speed

    obs = np.array([
        d_alt / 3000.0,
        d_hdg,
        d_spd / 100.0,
        alt / 5000.0,
        np.sin(roll),
        np.cos(roll),
        np.sin(pitch),
        fwd_speed / 340.0,
        side_speed / 50.0,
        vert_speed / 50.0,
        ias / 340.0,
        roll_rate / np.pi,
        pitch_rate / 1.0,
        yaw_rate / 0.5,
    ], dtype=np.float32)
    return np.clip(obs, -10.0, 10.0)

class LowLevel:
    '''frozen controller, get back stick bins'''

    def __init__(self, tag="controller_v1.0.1"):
        self.model = PPO.load(tag + ".zip", device="cpu")

    def bins(self, ac, tgt_alt, tgt_hdg, tgt_spd):
        b, _ = self.model.predict(control_obs(ac, tgt_alt, tgt_hdg, tgt_spd), deterministic=True)
        return b