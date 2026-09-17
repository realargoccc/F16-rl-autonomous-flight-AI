import csv
import math
import numpy as np
from pisnap import SnapEnv
from selector import load_policy

#python csvtotacview.py snap_best.csv f16_snap_v1.0.2.acmi
TAG = "snap_v1.1.0"
BASELINE = "snap_v1.0.1"         
OPPONENT = "pursuit"               #"pursuit" or "v4.1.6"
REL_ALT_BAND = (-500.0, 500.0)
SEEDS = range(80000, 80060)   
W = 18                              #column width 

def sweep(tag):
    model, rms, clip, eps = load_policy(tag)
    env = SnapEnv(opponent=OPPONENT, rel_alt_band = REL_ALT_BAND)                      #fresh env per policy
    f16 = env.env
    passes = []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        t0 = f16.me.get_sim_time()
        foe_hp, own_hp = f16.foe_hp, f16.agent_hp
        offset = float(f16.foe.pos()[2] - f16.me.pos()[2])
        rows, on_target, closest, nose900 = [], 0, float("inf"), np.nan
        terminated = truncated = False
        while not (terminated or truncated):
            n = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip)
            action, _ = model.predict(n.astype(np.float32), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            closest = min(closest, f16.range)
            if np.isnan(nose900) and f16.range <= f16.gun_rmax:
                nose900 = math.degrees(f16.boresight)    
            if f16.gun_rmin <= f16.range <= f16.gun_rmax and math.degrees(f16.boresight) < 3.0:
                on_target += 1
            me, foe = f16.me.pos(), f16.foe.pos()

            rows.append({
                "time": f16.me.get_sim_time() - t0,
                "bank_rad": f16.me['attitude/phi-rad'],
                "pitch_rad": f16.me['attitude/theta-deg'],   
                "heading_deg": f16.me['attitude/psi-deg'],
                "agent_n_m": float(me[0]), "agent_e_m": float(me[1]), "agent_up_m": float(me[2]),
                "foe_n_m": float(foe[0]), "foe_e_m": float(foe[1]), "foe_up_m": float(foe[2]),
                "foe_roll_deg": f16.foe['attitude/phi-deg'],
                "foe_pitch_deg": f16.foe['attitude/theta-deg'],
                "foe_yaw_deg": f16.foe['attitude/psi-deg'],
            })
        if info["win"]:                    end = "kill"
        elif f16.agent_hp <= 0.0:          end = "shot down"
        elif info["deck_hit"] or info["crashed"]: end = "own over-g/deck"
        elif info["foe_down"]:             end = "foe down"
        elif info["pass_done"]:            end = "pass"
        else:                              end = "no pass"
        passes.append({"seed": seed, "end": end, "on_target": on_target, "closest": closest,
                       "dealt": foe_hp - f16.foe_hp, "taken": own_hp - f16.agent_hp, "rows": rows
                       "nose900": nose900})
    return passes
    
def summarize(passes):
    get_stats = lambda k: [p[k] for p in passes]
    ends = get_stats("end")
    return {"passes":     len(passes),
            "hit passes": sum(n > 0 for n in get_stats("on_target")),
            "on target":  np.mean(get_stats("on_target")),
            "dealt":      np.mean(get_stats("dealt")),
            "taken":      np.mean(get_stats("taken")),
            "nose@900":   np.nanmean(get_stats("nose900")),
            "kills":      ends.count("kill"),
            "lost":       ends.count("shot down") + ends.count("own over-g/deck"),
            "closest m":  np.median(get_stats("closest"))}

if __name__ == "__main__":
    base, snap = sweep(BASELINE), sweep(TAG)
    sb, ss = summarize(base), summarize(snap)
    print(f"{OPPONENT}, {len(SEEDS)} passes {BASELINE:>13}{TAG:>13}")
    for k in sb:
        print(f"{k:12}{sb[k]:>13.3g}{ss[k]:>13.3g}")

    best = max(snap, key=lambda p: (p["dealt"], p["on_target"]))
    with open("snap_best.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=best["rows"][0].keys())
        w.writeheader()
        w.writerows(best["rows"])
    print("wrote snap_best.csv  seed", best["seed"], best["end"])