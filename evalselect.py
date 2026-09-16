import csv
import numpy as np
from flight_env import F16Env
from selector import Selector 
from piselect import LearnedSelector

TAG = "select_v1.0.0"
SEEDS = range(30000, 30060)       #merge ruler, never change
ENDS = ("kill", "shot down", "own over-g", "own deck", "foe over-g", "foe deck", "timeout")

def make_env(opponent):
    env = F16Env()
    env.aspect_band, env.range_band = (150.0, 180.0), (2500.0, 4000.0)
    env.defensive_p = 0.0
    if opponent == "v4.1.6":
        env.load_foe("v4.1.6")
    env.foe_pool_prob = 1.0 if opponent == "v4.1.6" else 0.0
    return env

def fight(env, agent, sel, seed):
    obs, _ = env.reset(seed=seed)
    agent.reset()
    t0 = env.me.get_sim_time()
    rows = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = agent.predict(obs)
        flown = "defense" if sel.quadrant(obs) == "defense" else agent.mode   #the expert that actually flew
        obs, reward, terminated, truncated, info = env.step(action)
        me, foe = env.me.pos(), env.foe.pos()
        rows.append({
            "time": env.me.get_sim_time() - t0,
            "bank_rad": env.me['attitude/phi-rad'],
            "pitch_rad": env.me['attitude/theta-deg'],   #degrees
            "heading_deg": env.me['attitude/psi-deg'],
            "agent_n_m": float(me[0]), "agent_e_m": float(me[1]), "agent_up_m": float(me[2]),
            "foe_n_m": float(foe[0]), "foe_e_m": float(foe[1]), "foe_up_m": float(foe[2]),
            "foe_roll_deg": env.foe['attitude/phi-deg'],
            "foe_pitch_deg": env.foe['attitude/theta-deg'],
            "foe_yaw_deg": env.foe['attitude/psi-deg'],
            "mode": flown,                               #shows key essential for tacview to generate
        })
    if info["win"]:                                   end = "kill"
    elif env.agent_hp <= 0.0:                         end = "shot down"
    elif info["deck_hit"]:                            end = "own deck"
    elif info["crashed"]:                             end = "own over-g"
    elif info["foe_down"] and info["foe_crashed"]:    end = "foe over-g"
    elif info["foe_down"]:                            end = "foe deck"
    else:                                             end = "timeout"
    return {"seed": seed, "end": end, "win": end == "kill", "dealt": 1.0 - float(env.foe_hp),
            "length": len(rows), "rows": rows}

def sweep(agent_kind, opponent):    #agent vs opponent on all 60 seeds
    env = make_env(opponent)
    sel = Selector()
    agent = sel if agent_kind == "hand" else LearnedSelector(TAG, sel)
    return [fight(env, agent, sel, s) for s in SEEDS]

def summarize(episodes):    #print all 60 seeds in one column
    frames = [r["mode"] for epi in episodes for r in epi["rows"]]
    s = {end: sum(epi["end"] == end for epi in episodes) for end in ENDS}
    for m in Selector.MODES:
        s[m] = 100.0 * frames.count(m) / len(frames)
    return s

if __name__ == "__main__":
    for opponent in ("bandit", "v4.1.6"):
        print("hand    vs", opponent, summarize(sweep("hand", opponent)))
        learned = sweep("learned", opponent)
        print("learned vs", opponent, summarize(learned))
        best = max(learned, key=lambda e: (e["win"], e["dealt"], -e["length"]))   #priority: kill, damage, shorter
        path = "select_best_" + opponent.replace(".", "") + ".csv"

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=best["rows"][0].keys())
            writer.writeheader()
            writer.writerows(best["rows"])
        print("wrote", path, "seed", best["seed"], best["end"])

#python csvtotacview.py select_best_bandit.csv f16_select_v1.0.0_bandit.acmi
#python csvtotacview.py select_best_v416.csv f16_select_v1.0.1_v416.acmi
