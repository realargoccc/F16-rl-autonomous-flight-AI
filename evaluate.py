'''
    Objective: Among all the trainings runs, pick the best performed run, pick the value and convert it to CSV file to check data accuracy.

    Purpose: Understand that the agent is flying normal not just learning for rewards

    How: Focus on total steps completed, highest reward, explained variance closer to 1

    Priority: Steps, Reward, explained variance

'''
import os
from flight_env import F16Env
from flight_env import F16Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import jsbsim
import csv
import math

ROOT = os.path.join(os.path.dirname(__file__), "jsbsim-data")
#fdm = jsbsim.FGFDMExec(ROOT, None)

vecnorm_path = "vecnorm_eleva_v1.9.1.pkl"
tmp = DummyVecEnv([lambda: F16Env()])
vecnorm = VecNormalize.load(vecnorm_path, tmp)
vecnorm.training = False           #Freeze stats during eval
vecnorm.norm_reward = False
model = PPO.load("ppo_f16_eleva_v1.9.1.zip")
raw = F16Env()

def get_episode(model, vecnorm, raw):
    obs, _ = raw.reset()     #reset observations
    start_time = raw.fdm.get_sim_time()
    total_reward = 0
    rows = []                   #create storage for later transition to CSV content
    step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        norm_obs = vecnorm.normalize_obs(obs)
        action, _ = model.predict(norm_obs, deterministic=True) #given current observation, what will the model do? / True means use model's preferred action

        obs, reward, terminated, truncated, info = raw.step(action)     #applies the action, and returns the five
        total_reward += float(reward)

        rows.append({
            "time": raw.fdm.get_sim_time() - start_time,
            "lat_deg":  raw.fdm['position/lat-geod-deg'],
            "lon_deg":  raw.fdm['position/long-gc-deg'],
            "alt_msl_m": raw.fdm['position/h-sl-meters'],
            "pitch_rad":  raw.fdm['attitude/theta-deg'], #below 3 value's rad are deg, set rad to match the analyzer unit
            "bank_rad": raw.fdm['attitude/phi-rad'],
            "yaw_angle": raw.fdm['aero/beta-deg'],      #sideslip in degrees
            "yaw_rate": raw.fdm['velocities/r-rad_sec'],
            "turn_rate": raw.fdm['velocities/psidot-rad_sec'] * 57.2958, #check for min radius turn 
            "heading_deg": raw.fdm['attitude/psi-deg'],
            "turned_deg": np.degrees(raw.turned),
            "vx_ms": raw.fdm['velocities/v-north-fps'] * 0.3048,
            "vy_ms": raw.fdm['velocities/v-east-fps'] * 0.3048,
            "vz_ms": raw.fdm['velocities/v-down-fps'] * 0.3048,
            "ias_ms": raw.fdm['velocities/vc-fps'] * 0.3048,
            "engine_n1": raw.fdm['propulsion/engine/n1'],
            "engine_n2": raw.fdm['propulsion/engine/n2'],
            "thrust_lbs": raw.fdm['propulsion/engine/thrust-lbs'],
            "mach": raw.fdm['velocities/mach'],
            "aoa_rad": raw.fdm['aero/alpha-deg'],   #csv analyzer takes deg, naming rad to match analyzer unit
            "g_load": raw.fdm['accelerations/Nz'], #aircraft g, pilot g are /n-pilot-z-norm
            "vertical_speed_ms": raw.fdm['velocities/h-dot-fps'] * 0.3048, 
            "engine_rpm_left": 0.0,     #f16 only has one engine so only one engine data record - also engine rpm is irrelevant to RL
            "engine_rpm_right": 0.0,
            "fuel_internal": 0.0,       #fuel is not important at this stage
            "gear_pos": raw.fdm['gear/gear-pos-norm'], # 0 - 1
            "alt_agl_m": raw.fdm['position/h-agl-ft'] * 0.3048,
            #above are csv format, below are additional checkings
            "step": step,
            "reward": reward,
            "cumulative_reward": total_reward,
            "done": bool(terminated or truncated),
            "throttle": raw.fdm['fcs/throttle-cmd-norm'],
            "elevator": raw.fdm['fcs/elevator-cmd-norm'],
            "aileron": raw.fdm['fcs/aileron-cmd-norm'],
            "rudder" : raw.fdm['fcs/rudder-cmd-norm']
            }
        )
        step += 1
    completed = bool(abs(raw.turned) >= 2 * np.pi)  #turned 360 degrees are consider completed
    summary = {
        "length": step,
        "total_reward": total_reward,
        "completed": completed,
        "turned_deg": float(np.degrees(abs(raw.turned))),
        "rows": rows,
    }
    return summary

def episode_key(epi):
    if epi["completed"]:    #if this episode completed turn. 
        return (1, -epi["length"], epi["total_reward"])
    return (0, epi["turned_deg"], epi["total_reward"])


def get_best_epi (model, vecnorm, raw, num_episodes):
    best_run = None
    for _ in range(num_episodes):
        episode = get_episode(model, vecnorm, raw)
        if best_run is None or episode_key(episode) > episode_key(best_run):
            best_run = episode

    return best_run

peak_episode = get_best_epi(model, vecnorm, raw, num_episodes=50) #pick the best episode first
print(f"best episode -> completed: {peak_episode["completed"]} steps: {peak_episode["length"]}, turned deg: {peak_episode["turned_deg"]}, total reward: {peak_episode["total_reward"]}")
field_names = list(peak_episode["rows"][0].keys())
with open ("eval_best.csv", "w", newline="") as f:     #open the csv 
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    #writer.writerows(peak_episode["rows"]) Option A:
    for row in peak_episode["rows"]:        #Option B: I prefer B as it is detailed 
        writer.writerow(row)

#run code: python evaluate.py
# fix best-episode comparison
# use CSV writer or manually convert each row to comma-separated 



