'''
    Objective: one fixed way to measure a checkpoint, so two numbers can actually be compared.

    Purpose: every time the env changes something silently the win rate moves and I dont find out
    for days. This runs the same 180 fights every time on the same seeds, so if the number moves
    it is either the policy or the env, not the dice.

    How: two seed blocks (9000 and 20000), report per cell not aggregate, paired McNemar when
    comparing two checkpoints.

    Rule: dont change the seeds or the cell definition. If they change all the old numbers are dead.
'''
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from flight_env import F16Env 

model_version = "v2.8.8"
compare_vers = "v2.8.7"
baseline = 0.83         #last seen good version
tol = 0.05              #how far it can drift

seeds = list(range(9000, 9090)) + list(range(20000, 20090))
aspect_edges = [20.0, 40.0, 60.0, 80.0]
aspect_names = ["0-20", "20-40", "40-60", "60-80"]
tactics = ["flee", "beam+", "beam-"]
pitches = ["climb", "level", "dive"]

def run_sweep(tag):
    model = PPO.load("ppo_f16_eleva_" + tag + ".zip")
    tmp = DummyVecEnv([lambda: F16Env()])
    vecnorm = VecNormalize.load("vecnorm_eleva_" + tag + ".pkl", tmp)
    vecnorm.training = False
    vecnorm.norm_reward = False
    raw = F16Env()
    rows = []

    for s in seeds:
        obs, info = raw.reset(seed=s)
        if raw.turn_offset > 2.0: tac = "flee"
        elif raw.turn_offset > 0: tac = "beam+"
        else: tac = "beam-"

        if raw.pitch_target > 0.01: pit = "climb"
        elif raw.pitch_target < -0.01: pit = "dive"
        else: pit = "level"

        aa = abs(raw.spawn_aspect) #aspect angle
        asp = aspect_names[-1]
        for i in range(len(aspect_edges)):
            if aa < aspect_edges[i]:
                asp = aspect_names[i]
                break

        r0 = raw.range      #range at spawn
        rmin = r0
        dwell = 0           #steps in wez(range)
        bs_list = []        #boresight steps in wez(range)
        crashed = False
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, info = raw.step(action)

            if raw.range < rmin:
                rmin = raw.range
            if raw.gun_rmin <= raw.range <= raw.gun_rmax:
                dwell += 1
                bs_list.append(np.degrees(raw.boresight))
            if raw.me['position/h-agl-ft'] * 0.3048 < 30 or abs(raw.me['accelerations/Nz']) > 13.0:
                crashed = True
        if len(bs_list) > 0:    #if never enter wez, need a bad run (warning)
            medbs = float(np.median(bs_list))
        else:
            medbs = np.nan

        rows.append({"seed": s, "tac": tac, "pit": pit, "asp":asp,
                        "win": bool(raw.foe_hp <= 0.0), 
                        "crashed": crashed, 
                        "spawn_range": r0, 
                        "min_range": rmin,
                        "dwell": dwell,
                        "foe_hp": float(raw.foe_hp),
                        "medbs": medbs,})
    return rows

#helpers
def count_cell(rows, key, value):
    w = 0
    n = 0
    for r in rows:
        if r[key] == value:
            n += 1
            if r["win"]:
                w += 1
    return w, n

def count_pair(rows, tac, pit):
    w = 0
    n = 0
    for r in rows:
        if r["tac"] == tac and r["pit"] == pit:
            n += 1
            if r["win"]:
                w += 1
    return w, n

def display(ver, rows):
    wins = 0
    crashes = 0
    never_wez = 0
    for r in rows:
        if r["win"]: wins += 1
        if r["crashed"]: crashes += 1
        if r["dwell"] == 0: never_wez += 1
    n = len(rows)
    print("")
    print("=== " + ver + " ===  n=" + str(n))
    print(f"overall {wins}/{n} = {wins/n:.0%}   crashed {crashes/n:.0%}   timeout {(n-wins-crashes)/n:.0%}   never in wez {never_wez/n:.0%}")
    print("")
    header = "        "
    for p in pitches:
        header += f"{p:>10}"
    header += "     total"
    print(header)

    for t in tactics:
        line = f"{t:8}"
        for p in pitches:
            w, c = count_pair(rows, t, p)
            if c > 0:
                line += f"{str(w) + '/' + str(c):>10}"
            else:
                line += f"{'-':>10}"
        w, c = count_cell(rows, "tac", t)
        line += f"{str(w) + '/' + str(c):>10}"
        print(line)

    print("")
    line = "aspect  "
    for a in aspect_names:
        line += f"{a:>10}"
    print(line)
    line = "        "
    for a in aspect_names:
        w, c = count_cell(rows, "asp", a)
        if c > 0:
            line += f"{str(w) + '/' + str(c):>10}"
        else:
            line += f"{'-':>10}"
    print(line)

    #loss 
    losses = []
    for r in rows:
        if not r["win"]:
            losses.append(r)
    if len(losses) > 0:
        medbs = np.nanmean([r["medbs"] for r in losses])    #median boresight
        minr = np.mean([r["min_range"] for r in losses])
        dwell = np.mean([r["dwell"] for r in losses])
        hp = np.mean([r["foe_hp"] for r in losses])
        print("")
        print(f"losses n={len(losses)}:  medbs {medbs:.1f}deg   min_range {minr:.0f}m   dwell {dwell:.0f}   foe_hp {hp:.2f}")

#comparison between two runs
def show_pairs(rows_a, rows_b, ver_a, ver_b):
    a_only = 0
    b_only = 0
    for i in range(len(rows_a)):
        if rows_a[i]["win"] and not rows_b[i]["win"]:
            a_only += 1
        if rows_b[i]["win"] and not rows_a[i]["win"]:
            b_only += 1
    disc = a_only + b_only #case when one side win
    if disc > 0: #is the win difference significant
        diff_mag = (abs(a_only - b_only) - 1) ** 2 / disc
    else:
        diff_mag = 0.0
    print("")
    print("=== paired " + ver_a + " vs " + ver_b + " ===")
    print(f"{ver_a} only {a_only}   {ver_b} only {b_only}   McNemar chi2 = {diff_mag:.2f}   (3.84 = p<0.05)")
    print("")
    print(f"{'cell':16}{ver_a:>10}{ver_b:>10}")
    for a in aspect_names:
        wa, ca = count_cell(rows_a, "asp", a)
        wb, cb = count_cell(rows_b, "asp", a)
        if ca > 0:
            print(f"{'aspect ' + a:16}{str(wa) + '/' + str(ca):>10}{str(wb) + '/' + str(cb):>10}")
    for t in tactics:
        wa, ca = count_cell(rows_a, "tac", t)
        wb, cb = count_cell(rows_b, "tac", t)
        if ca > 0:
            print(f"{t:16}{str(wa) + '/' + str(ca):>10}{str(wb) + '/' + str(cb):>10}")

rows_main = run_sweep(model_version)
display(model_version, rows_main)

if compare_vers is not None:
    rows_other = run_sweep(compare_vers)
    display(compare_vers, rows_other)
    show_pairs(rows_main, rows_other, model_version, compare_vers)
else:
    wins = 0
    for r in rows_main:
        if r["win"]:
            wins += 1
    got = wins / len(rows_main)
    delta = got - baseline
    print("")
    if abs(delta) < tol:
        print(f"regression: {got:.0%} vs baseline {baseline:.0%} ({delta:+.1%})  PASS")
    else:
        print(f"regression: {got:.0%} vs baseline {baseline:.0%} ({delta:+.1%})  FAIL")