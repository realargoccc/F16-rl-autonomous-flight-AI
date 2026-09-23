"""setting up the environment for practice range testing of the mounted missile"""

import math
from stable_baselines3.common.monitor import Monitor
from flight_env import F16Env

RANGE_BAND = (3000.0, 8000.0)
ASPECT_BAND = (0.0, 20.0)       #target starts roughly flying away from agent
REL_ALT_BAND = (0.0, 0.0)
STATIONS = ("left_wingtip", "right_wingtip")

class RangeEnv(F16Env):
    def __init__(
        self, range_band=RANGE_BAND, aspect_band=ASPECT_BAND, rel_alt_band=REL_ALT_BAND, stations=STATIONS,
        max_steps=600, ):
        super().__init__()

        if not stations:
            raise ValueError("range environment needs at least one store station")

        self.range_band = range_band
        self.aspect_band = aspect_band
        self.rel_alt_band = rel_alt_band
        self.bearing_spread = 5.0
        self.defensive_p = 0.0
        self.foe_pool_prob = 0.0
        self.max_episodes_steps = max_steps

        #read in the F16Env reset bridge
        self.captive_wingtip_stores = True
        self.store_stations = tuple(stations)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._hold_heading()
        info["stores"] = self._store_info()
        return obs, info

    def _hold_heading(self):
        """Keep the scripted target straight, level, and at fixed speed."""
        los = self.me.pos() - self.foe.pos()
        bearing = math.atan2(los[1], los[0])

        self.turn_offset = self.foe["attitude/psi-rad"] - bearing
        self.pitch_target = 0.0
        self.nominal_speed = 200.0

    def _store_info(self):
        """Serializable store data for range_eval and future ACMI output."""
        mounted_states = self.store_states()
        stores = []

        for station_id in self.store_stations:
            state = mounted_states.get(station_id)
            mounted = self.loadout.get(station_id)

            if state is None or mounted is None:
                continue

            pos = state["pos"]
            roll_deg, pitch_deg, yaw_deg = state["att"]

            stores.append({
                    "parent_id": "agent",
                    "station_id": station_id,
                    "store_name": mounted.spec.name,
                    "display_name": mounted.spec.display_name,
                    "tacview_type": mounted.spec.tacview_type,
                    "store_state": mounted.state.value,
                    "body_position_m": tuple(mounted.hardpoint.body_position_m),
                    "north_m": float(pos[0]),
                    "east_m": float(pos[1]),
                    "up_m": float(pos[2]),
                    "roll_deg": float(roll_deg),
                    "pitch_deg": float(pitch_deg),
                    "yaw_deg": float(yaw_deg),
                })
        return stores

    def step(self, action):
        self._hold_heading()
        obs, reward, terminated, truncated, info = super().step(action)

        info["stores"] = self._store_info()
        return obs, reward, terminated, truncated, info

def make_range_env(range_band=RANGE_BAND, aspect_band=ASPECT_BAND, rel_alt_band=REL_ALT_BAND, stations=STATIONS,): 
    env = RangeEnv(
        range_band=range_band,
        aspect_band=aspect_band,
        rel_alt_band=rel_alt_band,
        stations=stations,
    )

    return Monitor(env, info_keywords=(
            "crashed",
            "foe_crashed",
            "win",
            "deck_hit",
            "foe_down",
        ),
    )