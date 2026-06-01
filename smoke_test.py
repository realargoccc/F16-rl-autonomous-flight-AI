import jsbsim
from flight_frame import FlightFrame
from dataclasses import asdict
# Point JSBSim at the data folder cloned
ROOT = r"C:\Users\argoccc\Desktop\jsbsim-rl\jsbsim-data"

fdm = jsbsim.FGFDMExec(ROOT, None)
fdm.load_model('f16')

# Initial conditions: 10,000 ft altitude, 500 ft/s airspeed
fdm['ic/h-sl-ft'] = 10000.0
fdm['ic/vt-fps'] = 500.0
fdm['ic/throttle-cmd-norm'] = 0.8
fdm['ic/elevator-cmd-norm'] = 0.2
fdm.run_ic()
#fdm.print_property_catalog()
def frame_from_jsbsims(fdm) -> FlightFrame:
    frame = FlightFrame()
    frame.time_sec = fdm.get_sim_time()
    frame.lat_deg = fdm['position/lat-geod-deg']
    frame.lon_deg = fdm['position/long-gc-deg']
    frame.alt_msl_m = fdm['position/h-sl-meters']
    frame.pitch_rad = fdm['attitude/theta-deg'] #below 3 value's rad are deg, set rad to match the analyzer unit
    frame.bank_rad = fdm['attitude/phi-rad']
    frame.heading_rad = fdm['attitude/psi-rad']
    frame.vx_ms = fdm['velocities/v-north-fps'] * 0.3048
    frame.vy_ms = fdm['velocities/v-east-fps'] * 0.3048
    frame.vz_ms = fdm['velocities/v-down-fps'] * 0.3048
    frame.ias_ms = fdm['velocities/vc-fps'] * 0.3048
    frame.mach = fdm['velocities/mach']
    frame.aoa_rad = fdm['aero/alpha-deg']   #csv analyzer takes deg, naming rad to match analyzer unit
    frame.g_load = fdm['accelerations/Nz'] #aircraft g, pilot g are /n-pilot-z-norm
    frame.vertical_speed_ms = fdm['velocities/h-dot-fps'] * 0.3048
    frame.engine_rpm_left = 0.0     #f16 only has one engine so only one engine data record - also engine rpm is irrelevant to RL
    frame.engine_rpm_right = 0.0
    frame.fuel_internal = 0.0       #fuel is not important at this stage
    frame.gear_pos = fdm['gear/gear-pos-norm'] # 0 - 1
    frame.alt_agl_m = fdm['position/h-agl-ft'] * 0.3048
    #Below doesn't exist, but for future use, keeping them
    #frame.vt = fdm['velocities/vt-fps']
    #frame.throttle = fdm['fcs/throttle-cmd-norm']
    #frame.elevator = fdm['fcs/elevator-cmd-norm']
    return frame

# default hz is 120, tick / 120 = seconds
with open('RLconvert.csv', 'w') as o_file:
    o_file.write("time,lat,lon,alt_msl_m,pitch_rad,bank_rad,heading_rad,vx_ms,vy_ms,vz_ms,ias_ms,mach,aoa_rad,g_load,vertical_speed_ms,engine_rpm_left,engine_rpm_right,fuel_internal,gear_pos,alt_agl_m\n")
    for i in range(600):
        fdm['fcs/throttle-cmd-norm'] = 0.8
        fdm['fcs/elevator-cmd-norm'] = -0.2
        fdm.run()
        frame = frame_from_jsbsims(fdm)

        values = asdict(frame).values()     #only want the value
        o_file.write(",".join(str(v) for v in values) + "\n")   #combine each value together in one string parsing by ","
        if i % 120 == 0:  # print once per simulated second
            print(frame)

