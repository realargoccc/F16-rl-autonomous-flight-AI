import csv
import math

csv_path = "eval_best.csv"
acmi_path = "f16_intercept_v2.1.9.acmi"

#nellis afb lat and lon (fight location)
ref_lat = 36.20
ref_lon = -115.00

reference_time = "2026-07-12T12:00:00Z" # fight time in UTC

agent_ID = "A0"
bandit_ID = "B0"

m_per_deg_lat = 111320.0

def local_to_lonlat(north_m, east_m):
    lat = ref_lat + north_m / m_per_deg_lat
    lon = ref_lon + east_m / (m_per_deg_lat * math.cos(math.radians(ref_lat)))
    return lon, lat

def fnum(x):
    return f"{x:.7f}".rstrip("0").rstrip(".")   #compact digits for cosmetics

def main():
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    
    t_0 = float(rows[0]["time"])

    output = [
        "FileType=text/acmi/tacview",
        "FileVersion=2.2",
        f"0,ReferenceTime={reference_time}",
        "0,DataSource=JSBSim + Stablebaseline3 PPO",
        "0,Author=F16-rl-autonomous-flight-AI",
    ]

    for i, r in enumerate(rows):
        t = float(r["time"]) - t_0
        output.append(f"#{t:.2f}")

        #agent stats
        a_lon, a_lat = local_to_lonlat(float(r["agent_n_m"]), float(r["agent_e_m"]))
        a_alt = float(r["agent_up_m"])
        roll = math.degrees(float(r["bank_rad"]))
        pitch = float(r["pitch_rad"])
        yaw = float(r["heading_deg"])
        T = f"{fnum(a_lon)}|{fnum(a_lat)}|{fnum(a_alt)}|{fnum(roll)}|{fnum(pitch)}|{fnum(yaw)}" 

        if i == 0:
            output.append(f"{agent_ID},T={T},Name=F-16C, "
                          f"Color=Blue,Callsign=Sheppherd,Pilot=V2.1.9")
        else:
            output.append(f"{agent_ID},T={T}")

        #bandit 
        b_lon, b_lat = local_to_lonlat(float(r["bandit_n_m"]), float(r["bandit_e_m"]))
        b_alt = float(r["bandit_up_m"])

        T_bandit = f"{fnum(b_lon)}|{fnum(b_lat)}|{fnum(b_alt)}" #0 0 0 in current model

        if i == 0: 
            output.append(f"{bandit_ID},T={T_bandit},Name=Su-27, "
                          f"Color=Red,Callsign=Bandit")
        else:
            output.append(f"{bandit_ID},T={T_bandit}")

    with open(acmi_path, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(output) + "\n")
    duration = float(rows[-1]["time"]) - t_0
    print(f"Wrote {acmi_path}")
    print(f"  {len(rows)} frames, {duration:.1f}s, location {ref_lat},{ref_lon}")

if __name__ == "__main__":
    main()