import time
from flight_env import F16Env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

model_path = "ppo_f16_eleva_v2.0.9.zip"
vecnorm_path = "vecnorm_eleva_v2.0.9.pkl"
fg_directive = "data_output/flightgear.xml" #sends net_fdm to localhost:5550 (receiver in flightgear settings)

env = DummyVecEnv([lambda: F16Env()])
env = VecNormalize.load(vecnorm_path, env)
env.norm_reward = False                 #DO NOT vecnorm reward
env.training = False                    #
model = PPO.load(model_path, env=env)

raw_env = env.venv.envs[0]  #Only vecnorm for the low variance train, still want original data for real time displaying 
fdm = raw_env.fdm           #getting the O.G. raw data 

#turn on  flightgear udp output, if this errors on the path, try the 
#root-relative form instead: fdm.set_output_directive("data_output/flightgear.xml")
fdm.set_output_directive(fg_directive) #pass the raw data to flightgear
fdm.enable_output()                     

#Sleep every update otherwise the plane will just teleport to the destination 
dt = fdm.get_delta_t()      #0.00833 s -> 120 Hz sim rate: dt = delta t, get_delta_t tells you it reports to whatever hz configured (this case: 120)
step_dt = dt * raw_env.sim_steps_per_action
obs = env.reset()       #set the scenes, ready for loop
print(f"current frame: {1/dt:.0f} Hz, switch to flightgear to watch flight live")

try:
    while True:   
        loop_start = time.perf_counter()    #count the initial time

        action, _ = model.predict(obs, deterministic=True)  #Decide action
        obs, rewards, done, infos = env.step(action)        #Apply action

        sleep_t = step_dt - (time.perf_counter() - loop_start)   #remaining sim time in comparison to real life time
        if sleep_t > 0:                                     #pause the remaining sim time to match real life time
            time.sleep(sleep_t)
except KeyboardInterrupt:
    print("\nFlight stopped")

'''
add to the flightgear settings args to ensure cockpit instruments are on
--timeofday=noon
--prop:/controls/electric/battery-switch=true
--prop:/controls/electric/external-power=true
--prop:/controls/engines/engine[0]/generator=true
--prop:/controls/electric/avionics-switch=true
-prop:/sim/model/autostart=true
'''