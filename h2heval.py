import os
from flight_env import F16Env
from pih2h import make_h2h_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import jsbsim
import csv
import math

ROOT = os.path.join(os.path.dirname(__file__), "jsbsim-data")

TAG = "h2h_v1.0.1"
FOE = "h2h_v1.0.1_g05"      #which generation to fly against

vecnorm_path = "vecnorm_eleva_" + TAG + ".pkl"
tmp = DummyVecEnv([lambda: F16Env()])
vecnorm = VecNormalize.load(vecnorm_path, tmp)
vecnorm.training = False           #Freeze stats during eval
vecnorm.norm_reward = False
model = PPO.load("ppo_f16_eleva_" + TAG + ".zip")
raw = make_h2h_env(pool=(FOE,)).env

def get_episode(model, vecnorm, raw, seed=None):
    obs, _ = raw.reset(seed=seed)     #reset observations
    rel_alt0 = float(raw.foe.pos()[2] - raw.me.pos()[2])
    spawn_aspect = float(raw.spawn_aspect)
    foe_tag = raw.foe_tags[raw.foe_idx] if raw.foe_idx >= 0 else "bandit"
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
        action, _ = model.predict(norm_obs, deterministic=True) #given current observation, what will the model do?
        obs, reward, terminated, truncated, info = raw.step(action)     #applies the action, and returns the five
        min_range = min(min_range, raw.range)
        min_off_angle = min(min_off_angle, np.degrees(raw.boresight))
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
            "engine_rpm_left": 0.0,     #f16 only has one engine so only one engine data record
            "engine_rpm_right": 0.0,
            "fuel_internal": 0.0,       #fuel is not important at this stage
            "gear_pos": raw.me['gear/gear-pos-norm'], # 0 - 1
            "alt_agl_m": raw.me['position/h-agl-ft'] * 0.3048,
            #intercept metrics
            "range_nm": raw.range / 1852.0,
            "boresight_az_deg" : np.degrees(raw.boresight_az),
            "closure_ms": float(raw.closure),
            "in_wez": bool (raw.gun_rmin <= raw.range <= raw.gun_rmax),
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
            "rudder" : raw.me['fcs/rudder-cmd-norm'],
            }
        )
        step += 1
    win = bool(raw.foe_hp <= 0.0)
    lose = bool(raw.agent_hp <= 0.0)
    crashed = bool(info["crashed"] or info["deck_hit"])
    alt_lost = float(rows[0]["alt_msl_m"] - min(r["alt_msl_m"] for r in rows))
    wez_bs = [abs(r["boresight_az_deg"]) for r in rows if r["in_wez"]]
    mean_abs_bs = float(np.mean(wez_bs)) if wez_bs else 999.0
    summary = {
        "length": step,
        "total_reward": total_reward,
        "win": win,
        "lose": lose,
        "crashed": crashed,
        "foe_down": bool(info["foe_down"]),
        "foe_hp": float(raw.foe_hp),
        "agent_hp" : float(raw.agent_hp),
        "rel_alt_init": rel_alt0,
        "rows": rows,
        "spawn_aspect": spawn_aspect,
        "foe_tag": foe_tag,
        "alt_lost": alt_lost,
        "min_bs": min_off_angle,
        "min_range": min_range,
        "mean_abs_bs": mean_abs_bs,
        "dwell": len(wez_bs)
    }
    return summary

def episode_key(epi):
    #priority rank: a real kill, then how close the nose ever got, then survived long
    return (int(epi["win"]),
            -epi["min_bs"],
            epi["length"])

def failure_key(epi):
    #the fastest way it died is the most instructive tape
    return (-epi["length"], epi["alt_lost"])

def seed_sweep(model, vecnorm, raw, num_episodes=50):
    wins = 0        #total kills
    ground = 0      #how many ended with someone in the dirt
    episodes = []
    for epi in range(num_episodes):
        episode = get_episode(model, vecnorm, raw, seed=1000+epi)

        episodes.append(episode)
        if episode["win"]:
            wins += 1
        if episode["crashed"] or episode["foe_down"]:
            ground += 1

        end = ("WIN" if episode["win"] else "killed" if episode["lose"]
               else "crash" if episode["crashed"] else "foe_down" if episode["foe_down"] else "draw")
        print(f"ep{epi:02d} {episode['length']:5d} steps | {end:8} | "
              f"min bs {episode['min_bs']:5.1f} | dwell {episode['dwell']:4d} | "
              f"alt lost {episode['alt_lost']:5.0f}m | "
              f"reward={episode['total_reward']:8.1f}")

    aspects = {"0-60": [0, 0], "60-120": [0, 0], "120-180": [0, 0]}
    foes = {}
    for e in episodes:
        aa = abs(e["spawn_aspect"])
        k = "0-60" if aa < 60 else ("60-120" if aa < 120 else "120-180")
        aspects[k][0] += int(e["win"]); aspects[k][1] += 1
        foes.setdefault(e["foe_tag"], [0, 0])
        foes[e["foe_tag"]][0] += int(e["win"]); foes[e["foe_tag"]][1] += 1

    for group in (aspects, foes):
        for name, (w, n) in group.items():
            if n:
                print(f" {name:>18}: {w}/{n} = {w/n:.0%}")

    mean_alt = float(np.mean([e["alt_lost"] for e in episodes]))
    mean_len = float(np.mean([e["length"] for e in episodes]))
    print(f"\nwin rate: {wins} / {num_episodes} = {wins/num_episodes:.0%}   "
          f"ground impacts: {ground} / {num_episodes}")
    print(f"mean episode {mean_len:.0f} steps, mean altitude given up {mean_alt:.0f} m")

    return episodes

episodes = seed_sweep(model, vecnorm, raw, num_episodes=50)
best = max(episodes, key=episode_key)

field_names = list(best["rows"][0].keys())
with open ("h2h_best.csv", "w", newline="") as f:     #open the csv
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    for row in best["rows"]:
        writer.writerow(row)
print(f"best: min bs {best['min_bs']:.1f}  dwell {best['dwell']}  len {best['length']}  alt lost {best['alt_lost']:.0f}m")

losses = [e for e in episodes if e["lose"] or e["crashed"]]
if losses:
    worst = max(losses, key=failure_key)
    with open("h2h_worst.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for row in worst["rows"]:
            writer.writerow(row)
    print(f"worst: min bs {worst['min_bs']:.1f}  len {worst['length']}  alt lost {worst['alt_lost']:.0f}m")

#run code: python h2heval.py
#then:     python csvtotacview.py h2h_best.csv f16_h2h_v1.0.1.acmi