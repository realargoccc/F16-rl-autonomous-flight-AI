import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import jsbsim
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "jsbsim-data")

class F16Env(gym.Env):
    def __init__(self):
        self.fdm = jsbsim.FGFDMExec(ROOT, None) #load FDM
        self.fdm.set_debug_level(0)             #remove banners of aircraft configurations (hundres loc)
        self.fdm.load_model('f16')              #load f16
        super().__init__()
        self.observation_space = Box(low=-np.inf, high = np.inf, shape=(14,), dtype = np.float32)    #set throttle and elevator lower and upper bound
        self.action_space = Box(low = np.array([-1.0, -1.0, -1.0, -1.0], dtype = np.float32),
                                high = np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32), dtype = np.float32)
        self.max_episodes_steps = 300
        self.curr_step = 0
        self.target_alt_ft = 10000.0
        self.sim_steps_per_action = 12

    def reset(self, seed=None, options = None): #IMPORTANT: make sure to reset any CONSUMABLE units, trims maybe in the future
        super().reset(seed=seed)
        self.fdm['ic/h-sl-ft'] = 10000.0#self.np_random.integers(8000, 12000) #randomize the starting position of the aircraft
        self.fdm['ic/vc-kts'] = self.np_random.integers(350,400)  #knots
        self.fdm['ic/throttle-cmd-norm'] = 0.5
        self.fdm['ic/elevator-cmd-norm'] = 0.0
        self.fdm["gear/gear-cmd-norm"] = 0.0
        self.fdm['propulsion/tank[0]/contents-lbs'] = 1500.0 #set initial fuel tank
        self.fdm['propulsion/tank[1]/contents-lbs'] = 1500.0
        self.fdm['propulsion/engine/set-running'] = 1.0      #Make sure the engine starts
        self.fdm['ic/phi-deg'] = 0.0 #wings level (no bank)
        self.fdm['ic/psi-true-deg'] = 0.0 # spawn due northing heading 000
        self.fdm.run_ic()
        #counter reset
        #self.fdm['simulation/do_simple_trim'] = 1  #one time solution before agent take over
        self.curr_step = 0
        self.prev_elev = 0.0
        self.prev_aile = 0.0 
        
        self.prev_heading = self.fdm['attitude/psi-rad']
        self.turned = 0.0   #accumulator
        self.prev_pitch_rate = 0.0
        obs = self._get_obs()   #contains the 8 observation data from def _get_obs
        info = {}
        return obs, info
    
    def _get_obs(self):
        return np.array(
            [self.fdm['position/h-sl-meters'],          #altitude
            self.fdm['velocities/vc-fps'] * 0.3048,     #IAS
            self.fdm['attitude/theta-rad'],             #pitch
            self.fdm['velocities/q-rad_sec'],           #pitch rate
            self.fdm['velocities/h-dot-fps'] * 0.3048,  #vertical speed
            self.fdm['aero/alpha-deg'],                 #aoa-deg
            self.fdm['attitude/phi-rad'],               #bank angle in radians
            self.fdm['velocities/p-rad_sec'],           #roll rate
            self.fdm['propulsion/engine/n1'],           #engine rpm (low lag responder to throttle)
            self.fdm['accelerations/Nz'],               #g_load
            self.fdm['velocities/mach'],                #corner speed monitor
            self.fdm['velocities/r-rad_sec'],           #yaw rate
            self.fdm['aero/beta-deg'],                  #sideslip (yaw angle)    
            2*np.pi - self.turned,          #remaining degrees need to be turned 
            ], dtype = np.float32
        )
    
    def step(self, action):
        self.fdm['fcs/throttle-cmd-norm'] = float ((action[0] + 1.0) / 2.0)   #assign value back to the self.action_space
        self.fdm['fcs/elevator-cmd-norm'] = float (action[1])
        self.fdm['fcs/aileron-cmd-norm'] = float (action[2])
        self.fdm['fcs/rudder-cmd-norm'] = float (action[3])
        self.fdm["gear/gear-cmd-norm"] = 0.0
        #run 
        for _ in range(self.sim_steps_per_action):
            self.fdm.run()
        #get obs
        obs = self._get_obs()

        self.curr_step += 1
        alt_agl_m = self.fdm['position/h-sl-ft'] * 0.3048
        completed = bool(abs(self.turned) >= 2*np.pi) 
        crashed = bool((alt_agl_m < 30) or abs(self.fdm['attitude/phi-rad']) > np.radians(100))
        terminated = crashed or completed
        truncated = bool(self.curr_step >= self.max_episodes_steps)
        curr_alt_ft = self.fdm['position/h-sl-ft']
        target_alt_ft = self.target_alt_ft
        speed_knots = self.fdm['velocities/vc-fps'] * 0.592484    #speed in knots
        curr_throttle = self.fdm['fcs/throttle-cmd-norm']
        #Turning Policy Units
        curr_heading = self.fdm['attitude/psi-rad'] 
        curr_bank = self.fdm['attitude/phi-deg'] 

        #Min radius turn units:
        roll_rate = self.fdm['velocities/p-rad_sec']     
        pitch_rate = self.fdm['velocities/q-rad_sec']  
        delta_turn = (curr_heading - self.prev_heading + np.pi) % (2*np.pi) - np.pi
        self.turned += delta_turn
        self.prev_heading = curr_heading

        #reward computations
        reward = -0.1   #per step++, reward += 0.1        
        if crashed: #crashed
            reward -= 100.0
        if completed: reward += 50.0
        #air speed policy  
        if speed_knots < 350:
            reward -= 0.05 * abs(speed_knots - 350)
        #elif speed_knots > 400:
        #    reward -= 0.05 * (speed_knots - 400)

        #throttle policy (max turn specific)
        reward -= 0.1 * abs(curr_throttle - 1.0)
        #Optimized reward unit
        alt_diff_ft = curr_alt_ft - target_alt_ft

        #Elevator saturation / smoothness limiter 
        delta_elev = abs(action[1] - self.prev_elev)
        reward -= 0.30 * delta_elev
        self.prev_elev = action[1]
        #Altitude policy
        #reward += 0.15 * max(0.0, 1.0 - abs(alt_diff_ft) / 100)
        reward -= min(0.5, abs(alt_diff_ft) / 500)

        #Turning Policy
        delta_aile = abs(action[2] - self.prev_aile)
        reward -= 0.30 * delta_aile
        self.prev_aile = action[2]
        reward += 5.0 * delta_turn #+ when turning the commanded way 
        
        #guide banking
        reward -= 0.5 * max(0.0, abs(curr_bank) - 84)
        reward -= 0.1 * abs(roll_rate)
        d_pitch_rate = abs(pitch_rate - self.prev_pitch_rate)
        reward -= 0.2 * d_pitch_rate
        self.prev_pitch_rate = pitch_rate
        #G_load policy: 
        
        info = {}
        return obs, float(reward), terminated, truncated, info    
        
if __name__ == "__main__":
    env = F16Env()
    obs, info = env.reset()
    print(f"Obs count: {obs.shape}")
    print(f"Initial obs: {obs}")

    #initial reward
    total_reward = 0.0
    episode_rewards = []

    for i in range(3000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward 

        if i % 100 == 0:        #per second feedback of reward and total reward
            print(f"Step {i}: alt_diff from obs, reward = {reward:.3f}, cumulative = {total_reward:.1f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}, termianted = {terminated}, truncated = {truncated}, total reward: {total_reward:.1f}")
            episode_rewards.append(total_reward)
            total_reward = 0.0
            obs, info = env.reset()

    print(f"Wrapper validation completed, reward:{reward}")
#run command: python flight_env.py
#interpretor select command: /Users/y/Desktop/jsbsim-rl/.venv/bin/pythons
#commit and push command: git add -A, git commit -m "message", git push
#pull from pc: git fetch origin, git reset --hard origin/main