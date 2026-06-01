'''
    Objective: Among all the trainings runs, pick the best performed run, pick the value and convert it to CSV file to check data accuracy.

    Purpose: Understand that the agent is flying normal not just learning for rewards

    How: Focus on total steps completed, highest reward, explained variance closer to 1

    Priority: Steps, Reward, explained variance

'''
import os
from flight_env import F16Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import jsbsim
import csv

ROOT = os.path.join(os.path.dirname(__file__), "jsbsim-data")
#fdm = jsbsim.FGFDMExec(ROOT, None)

vecnorm_path = "vecnorm_eleva_v1.2.0.pkl"
env = DummyVecEnv([lambda: F16Env()])
env = VecNormalize.load(vecnorm_path, env)
env.training = False           #Freeze stats during eval
env.norm_reward = False
model = PPO.load("ppo_f16_eleva_v1.2.0.zip", env=env)
raw_env = env.venv.envs[0]



def get_episode(model, env, raw_env, max_step):
    obs = env.reset()     #reset observations
    start_time = raw_env.fdm.get_sim_time()
    total_reward = 0
    rows = []                   #create storage for later transition to CSV content

    for step in range(max_step):
        action, _ = model.predict(obs, deterministic=True) #given current observation, what will the model do? / True means use model's preferred action

        obs, rewards, dones, infos = env.step(action)     #applies the action, and returns the five
        reward = float(rewards[0])
        done = bool(dones[0])

        total_reward += reward
        rows.append({
            "time": raw_env.fdm.get_sim_time() - start_time,
            "lat_deg":  raw_env.fdm['position/lat-geod-deg'],
            "lon_deg":  raw_env.fdm['position/long-gc-deg'],
            "alt_msl_m": raw_env.fdm['position/h-sl-meters'],
            "pitch_rad":  raw_env.fdm['attitude/theta-deg'], #below 3 value's rad are deg, set rad to match the analyzer unit
            "bank_rad": raw_env.fdm['attitude/phi-rad'],
            "heading_rad": raw_env.fdm['attitude/psi-rad'],
            "vx_ms": raw_env.fdm['velocities/v-north-fps'] * 0.3048,
            "vy_ms": raw_env.fdm['velocities/v-east-fps'] * 0.3048,
            "vz_ms": raw_env.fdm['velocities/v-down-fps'] * 0.3048,
            "ias_ms": raw_env.fdm['velocities/vc-fps'] * 0.3048,
            "engine_n1": raw_env.fdm['propulsion/engine/n1'],
            "engine_n2": raw_env.fdm['propulsion/engine/n2'],
            "thrust_lbs": raw_env.fdm['propulsion/engine/thrust-lbs'],
            "mach": raw_env.fdm['velocities/mach'],
            "aoa_rad": raw_env.fdm['aero/alpha-deg'],   #csv analyzer takes deg, naming rad to match analyzer unit
            "g_load": raw_env.fdm['accelerations/Nz'], #aircraft g, pilot g are /n-pilot-z-norm
            "vertical_speed_ms": raw_env.fdm['velocities/h-dot-fps'] * 0.3048, 
            "engine_rpm_left": 0.0,     #f16 only has one engine so only one engine data record - also engine rpm is irrelevant to RL
            "engine_rpm_right": 0.0,
            "fuel_internal": 0.0,       #fuel is not important at this stage
            "gear_pos": raw_env.fdm['gear/gear-pos-norm'], # 0 - 1
            "alt_agl_m": raw_env.fdm['position/h-agl-ft'] * 0.3048,
            #above are csv format, below are additional checkings
            "step": step,
            "reward": reward,
            "cumulative_reward": total_reward,
            "done": dones,
            "throttle": raw_env.fdm['fcs/throttle-cmd-norm'],
            "elevator": raw_env.fdm['fcs/elevator-cmd-norm']
            }
        )

        if dones:     #avoid stepping over the terminated episode
            break
    summary = {
        "length": step + 1,
        "total_reward": total_reward,
        "rows": rows
    }
    return summary

def get_best_epi (model, env, raw_env, max_step, num_episodes):
    peak_episode = None
    i = 1
    for i in range(num_episodes):
        episode = get_episode(model, env, raw_env, max_step)
        if peak_episode is None:        #first iteration when the peak is not assigned
            peak_episode = episode
        elif episode["length"] > peak_episode["length"]:    #first priority: training with longer steps wins
            peak_episode = episode
        elif episode["length"] == peak_episode["length"] and episode["total_reward"] > peak_episode["total_reward"]:      
            peak_episode = episode         #second priority: if steps are the same, then higher reward wins 

    return peak_episode

peak_episode = get_best_epi(model, env, raw_env, max_step=3600, num_episodes=50) #pick the best episode first
field_names = ["time","lat_deg","lon_deg","alt_msl_m","pitch_rad","bank_rad","heading_rad","vx_ms","vy_ms","vz_ms","ias_ms","engine_n1","engine_n2","thrust_lbs","mach","aoa_rad","g_load","vertical_speed_ms","engine_rpm_left","engine_rpm_right","fuel_internal","gear_pos","alt_agl_m", "step", "reward", "cumulative_reward", "done", "throttle", "elevator"]
with open ("eval_best.csv", "w", newline="") as f:     #open the csv 
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    #writer.writerows(peak_episode["rows"]) Option A:
    for row in peak_episode["rows"]:        #Option B: I prefer B as it is detailed 
        writer.writerow(row)

#run code: python evaluate.py
# fix best-episode comparison
# use CSV writer or manually convert each row to comma-separated 
