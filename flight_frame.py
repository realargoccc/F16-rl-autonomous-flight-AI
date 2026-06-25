from dataclasses import dataclass
from enum import Enum

class Severity(Enum):
    NORMAL = 1
    CAUTIOUS = 2
    CRITICAL = 3
    CATASTROPHIC = 4
    
@dataclass
class FlightFrame:
    time_sec: float = 0.0 # Mission time in seconds since mission start

    # Position
    lat_deg: float = 0.0  # Latitude in degrees (double for precision)
    lon_deg: float = 0.0  # Longitude in degrees (double for precision)
    alt_msl_m: float = 0.0 # Altitude above mean sea level, meters

    # Attitude
    pitch_deg: float = 0.0   # Nose up/down, radians
    bank_rad: float = 0.0   # Roll left/right, radians
    heading_deg: float = 0.0 # Compass heading, radians
    targ_heading: float = 0.0 #target heading in degrees
    heading_error: float = 0.0 # diff of targ heading and curr heading

    # Velocity (world-frame components)
    vx_ms: float = 0.0
    vy_ms: float = 0.0
    vz_ms: float = 0.0

    # Airspeed and aerodynamics
    ias_ms: float = 0.0  # Indicated airspeed, m/s
    #Engine status
    engine_n1: float = 0.0      #how much thrust are producing
    engine_n2: float = 0.0      #Is the engine capable of producing thrust?
    thrust_lbs: float = 0.0     #F16 engine max estimate: 25,000 lbf
    mach: float = 0.0    # Mach number (dimensionless)
    aoa_rad: float = 0.0 # Angle of attack, radians

    # Loads
    g_load: float = 0.0 # Vertical G (note: currently buggy in Lua, fix later)
    vertical_speed_ms: float = 0.0

    # Engines and fuel
    engine_rpm_left: float = 0.0  # Percent (0-100)
    engine_rpm_right: float = 0.0 # Percent (0-100)
    fuel_internal: float = 0.0    # Fraction (0.0-1.0)

    # Gears and AGL for TAXI, TAKEOFF, and LANDING
    gear_pos: float = 0.0
    alt_agl_m: float = 0.0

    #additional RL values:
    step: float = 0.0
    reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    throttle: float = 0.0
    elevator: float = 0.0
    aileron: float = 0.0
    # float ground_speed = std::sqrt(vx_ms * vx_ms + vz_ms * vz_ms);  #ask before implement it

def parse_csv_line(line: str) -> FlightFrame | None:
    field_store = line.split(",") # parsed each line
    frame = FlightFrame()         # frame as a side split for FlightFrame
    #edge case
    if len(field_store) != 30:
        print("cannot parse this, something is wrong")
        return None

    try: 
        frame.time_sec = float(field_store[0])
        frame.lat_deg = float(field_store[1])
        frame.lon_deg = float(field_store[2])
        frame.alt_msl_m = float(field_store[3])
        frame.pitch_deg = float(field_store[4])
        frame.bank_rad = float(field_store[5])
        frame.heading_rad = float(field_store[6])
        frame.vx_ms = float(field_store[7])
        frame.vy_ms = float(field_store[8])
        frame.vz_ms = float(field_store[9])
        frame.ias_ms = float(field_store[10])
        frame.engine_n1 = float(field_store[11])
        frame.engine_n2 = float(field_store[12])
        frame.thrust_lbs = float(field_store[13])
        frame.mach = float(field_store[14])
        frame.aoa_rad = float(field_store[15])
        frame.g_load = float(field_store[16])
        frame.vertical_speed_ms = float(field_store[17])
        frame.engine_rpm_left = float(field_store[18])
        frame.engine_rpm_right = float(field_store[19])
        frame.fuel_internal = float(field_store[20])
        frame.gear_pos = float(field_store[21])
        frame.alt_agl_m = float(field_store[22])
        frame.step = float(field_store[23])
        frame.reward = float(field_store[24])
        frame.cumulative_reward = float(field_store[25])
        frame.done = bool(field_store[26])
        frame.throttle = float(field_store[27])
        frame.elevator = float(field_store[28])
        frame.aileron = float(field_store[29])
    except ValueError: 
        print(f"'{line}' includes a non ideal number")
        return None
    
    return frame

