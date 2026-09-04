from .base import BaseReward
import numpy as np

class Deck(BaseReward):
    def raw(self, env, computed):
        alt_km = computed.alt_agl_m / 1000.0
        sink_mach = -env.me['velocities/h-dot-fps'] * 0.3048 / 340.0 #positive climbing 

        pv = 0.0
        if alt_km <= env.safe_altitude:
            pv = -np.clip(sink_mach / env.k_sink * (env.safe_altitude - alt_km) / env.safe_altitude, 0.0, 1.0)

        ph = 0.0
        if alt_km <= env.danger_altitude:
            ph = np.clip(alt_km / env.danger_altitude, 0.0, 1.0) - 1.0 - 1.0

        return pv, ph