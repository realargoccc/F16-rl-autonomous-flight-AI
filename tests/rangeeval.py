import csv

from pirange import RangeEnv
from selector import Selector

SEED = 90000
OUT = "range_best.csv"
STORE_KEYS = ("north_m", "east_m", "up_m", "roll_deg", "pitch_deg", "yaw_deg")

def fly(seed=SEED):
    env = RangeEnv()
    sel = Selector()
    obs, _ = env.reset(seed=seed)
    sel.reset()
    t0 = env.me.get_sim_time()
    rows = []
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = sel.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        me, foe = env.me.pos(), env.foe.pos()
        row = {
            "time": env.me.get_sim_time() - t0,
            "bank_rad": env.me['attitude/phi-rad'],
            "pitch_rad": env.me['attitude/theta-deg'],   #degrees
            "heading_deg": env.me['attitude/psi-deg'],
            "agent_n_m": float(me[0]), "agent_e_m": float(me[1]), "agent_up_m": float(me[2]),
            "foe_n_m": float(foe[0]), "foe_e_m": float(foe[1]), "foe_up_m": float(foe[2]),
            "foe_roll_deg": env.foe['attitude/phi-deg'],
            "foe_pitch_deg": env.foe['attitude/theta-deg'],
            "foe_yaw_deg": env.foe['attitude/psi-deg'],
            "mode": sel.mode,
            "mach": env.me['velocities/mach'],
            "aoa_deg": env.me['aero/alpha-deg'],
            "foe_mach": env.foe['velocities/mach'],
            "foe_aoa_deg": env.foe['aero/alpha-deg'],
        }
        on_rail = {s["station_id"]: s for s in info["stores"]}
        for station_id in env.store_stations:        #same columns every row; an empty station writes mounted=0
            store = on_rail.get(station_id)
            row[f"{station_id}_mounted"] = int(store is not None)
            for key in STORE_KEYS:
                row[f"{station_id}_{key}"] = store[key] if store is not None else ""
        rows.append(row)
    return rows

if __name__ == "__main__":
    rows = fly()
    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    on = [k[:-len("_mounted")] for k in rows[-1] if k.endswith("_mounted") and rows[-1][k]]
    print(f"wrote {OUT}  seed {SEED}  {len(rows)} frames, {rows[-1]['time']:.1f}s, on the rails at the end: {on}")