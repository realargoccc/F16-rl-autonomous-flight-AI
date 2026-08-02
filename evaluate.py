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

vecnorm_path = "vecnorm_eleva_v2.4.3.pkl"
tmp = DummyVecEnv([lambda: F16Env()])
vecnorm = VecNormalize.load(vecnorm_path, tmp)
vecnorm.training = False           #Freeze stats during eval
vecnorm.norm_reward = False
model = PPO.load("ppo_f16_eleva_v2.4.3.zip")
raw = F16Env()

def get_episode(model, vecnorm, raw, seed=None):
    obs, _ = raw.reset(seed=seed)     #reset observations
    rel_alt0 = float(raw.bandit.pos[2] - raw.me.pos()[2])    #the alt diff, if < 0, nose down
    turn_offset = float(raw.bandit.turn_offset)
    start_time = raw.me.get_sim_time()
    total_reward = 0
    rows = []                   #create storage for later transition to CSV content
    step = 0
    terminated = truncated = False
    min_range = float("inf")
    min_off_angle = float("inf")
    steps_in_wez = 0
    while not (terminated or truncated):
        norm_obs = vecnorm.normalize_obs(obs)
        action, _ = model.predict(norm_obs, deterministic=True) #given current observation, what will the model do? / True means use model's preferred action
        obs, reward, terminated, truncated, info = raw.step(action)     #applies the action, and returns the five
        min_range =  min(min_range, raw.range)
        min_off_angle = min(min_off_angle, raw.boresight)
        if raw.gun_rmin <= raw.range <= raw.gun_rmax:
            steps_in_wez += 1

        total_reward += float(reward)

        rows.append({
            "time": raw.me.get_sim_time() - start_time,
            "lat_deg":  raw.me['position/lat-geod-deg'],
            "lon_deg":  raw.me['position/long-gc-deg'],
            "alt_msl_m": raw.me['position/h-sl-meters'],
            "pitch_rad":  raw.me['attitude/theta-deg'], #below 3 value's rad are deg, set rad to match the analyzer unit
            "bank_rad": raw.me['attitude/phi-rad'],
            "yaw_angle": raw.me['aero/beta-deg'],      #sideslip in degrees
            "yaw_rate": raw.me['velocities/r-rad_sec'],
            "turn_rate": raw.me['velocities/psidot-rad_sec'] * 57.2958, #check for min radius turn 
            "heading_deg": raw.me['attitude/psi-deg'],
            "turned_deg": np.degrees(raw.turned),
            "vx_ms": raw.me['velocities/v-north-fps'] * 0.3048,
            "vy_ms": raw.me['velocities/v-east-fps'] * 0.3048,
            "vz_ms": raw.me['velocities/v-down-fps'] * 0.3048,
            "ias_ms": raw.me['velocities/vc-fps'] * 0.3048,
            "engine_n1": raw.me['propulsion/engine/n1'],
            "engine_n2": raw.me['propulsion/engine/n2'],
            "thrust_lbs": raw.me['propulsion/engine/thrust-lbs'],
            "mach": raw.me['velocities/mach'],
            "aoa_rad": raw.me['aero/alpha-deg'],   #csv analyzer takes deg, naming rad to match analyzer unit
            "g_load": raw.me['accelerations/Nz'], #aircraft g, pilot g are /n-pilot-z-norm
            "vertical_speed_ms": raw.me['velocities/h-dot-fps'] * 0.3048, 
            "engine_rpm_left": 0.0,     #f16 only has one engine so only one engine data record - also engine rpm is irrelevant to RL
            "engine_rpm_right": 0.0,
            "fuel_internal": 0.0,       #fuel is not important at this stage
            "gear_pos": raw.me['gear/gear-pos-norm'], # 0 - 1
            "alt_agl_m": raw.me['position/h-agl-ft'] * 0.3048,
            #intercept metrics
            "range_nm": raw.range / 1852.0,
            "boresight_az_deg" : np.degrees(raw.boresight_az),
            "closure_ms": float(obs[-2]),
            "relative_alt_m" : float (raw.bandit.pos[2] - raw.me.pos()[2]),
            "in_wez": bool (raw.gun_rmin <= raw.range <= raw.gun_rmax),
            #bandit vs agent spatial track 
            "bandit_n_m": float(raw.bandit.pos[0]),
            "bandit_e_m": float(raw.bandit.pos[1]),
            "bandit_up_m": float(raw.bandit.pos[2]),
            "foe_n_m": float(raw.foe.pos()[0]),
            "foe_e_m": float(raw.foe.pos()[1]),
            "foe_up_m": float(raw.foe.pos()[2]),
            "foe_roll_deg": raw.foe['attitude/phi-deg'],
            "foe_pitch_deg": raw.foe['attitude/theta-deg'],
            "foe_yaw_deg": raw.foe['attitude/psi-deg'],
            "agent_n_m": float(raw.me.pos()[0]),
            "agent_e_m": float(raw.me.pos()[1]),
            "agent_up_m": float(raw.me.pos()[2]),
            #above are csv format, below are additional checkings
            "step": step,
            "reward": reward,
            "cumulative_reward": total_reward,
            "done": bool(terminated or truncated),
            "throttle": raw.me['fcs/throttle-cmd-norm'],
            "elevator": raw.me['fcs/elevator-cmd-norm'],
            "aileron": raw.me['fcs/aileron-cmd-norm'],
            "rudder" : raw.me['fcs/rudder-cmd-norm']
            }
        )
        step += 1
    completed = bool(abs(raw.turned) >= 2 * np.pi)  #turned 360 degrees are consider completed
    win = bool(raw.bandit.hp <= 0.0)
    lose = bool(raw.agent_hp <= 0.0)
    summary = {
        "length": step,
        "total_reward": total_reward,
        "win": win, 
        "lose": lose,
        "bandit_hp": float(raw.bandit.hp),
        "agent_hp" : float(raw.agent_hp),
        "rel_alt_init": rel_alt0,
        "rows": rows,
        "turn_offset": turn_offset,
    }
    return summary

def episode_key(epi):
    #priority rank: reached wez, closest distance to wez, nose point direction, reward
    return (int(epi["win"]),
            epi["agent_hp"] - epi["bandit_hp"],
            epi["total_reward"])

def seed_sweep(model, vecnorm, raw, num_episodes=50):
    wins = 0        #total kills
    low_wins = 0    #how many look down ends up a kill
    low_n = 0       #how many of 50 are look down cases
    episodes = []
    for epi in range(num_episodes):
        episode = get_episode(model, vecnorm, raw, seed=1000+epi)

        episodes.append(episode)
        if episode["win"]:
            wins += 1
        if episode["rel_alt_init"] < 0:
            low_n += 1
            low_wins += int(episode["win"])
        
        print(f"ep{epi:02d} relative alt: {episode['rel_alt_init']:+6.0f}meters | "
              f"win={str(episode['win']):5} | "
              f"a_hp:{episode['agent_hp']:.2f} | b_hp:{episode['bandit_hp']:.2f} | "
              f"reward={episode['total_reward']:7.1f} ")

    buckets = {"flee":[0,0], "beam +": [0,0], "beam -": [0,0]}
    for e in episodes:
        if e["turn_offset"] > 2.0: name = "flee"
        elif e["turn_offset"] > 0: name = "beam +"
        else: name = "beam -"
        buckets[name][0] += int(e["win"]); buckets[name][1] += 1
    for name, (w, n) in buckets.items():
        if n:
            print(f" {name:>7}: {w}/{n} = {w/n:.0%}")

    print(f"\nwin rate: {wins} / {num_episodes} = {wins/num_episodes:.0%}   "
          f"lower and win case: {low_wins} / {low_n}")
    
    return episodes

episodes = seed_sweep(model, vecnorm, raw, num_episodes=50)
best = max(episodes, key=episode_key)

field_names = list(best["rows"][0].keys())
with open ("eval_best.csv", "w", newline="") as f:     #open the csv 
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    #writer.writerows(peak_episode["rows"]) Option A:
    for row in best["rows"]:        #Option B: I prefer B as it is detailed 
        writer.writerow(row)

#run code: python evaluate.py
# fix best-episode comparison
# use CSV writer or manually convert each row to comma-separated 



