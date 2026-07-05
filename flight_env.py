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
        self.observation_space = Box(low=-np.inf, high = np.inf, shape=(21,), dtype = np.float32)    #set throttle and elevator lower and upper bound
        self.action_space = Box(low = np.array([-1.0, -1.0, -1.0, -1.0], dtype = np.float32),
                                high = np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32), dtype = np.float32)
        self.max_episodes_steps = 300
        self.curr_step = 0
        self.target_alt_ft = 10000.0
        self.sim_steps_per_action = 12

        #WEZ (Weapon Engagement Zone) configs
        nm_to_m = 1852.0
        self.rmax = 3.0 * nm_to_m  #
        self.rmin = 0.5 * nm_to_m
        self.rne = 1.0 * nm_to_m   #no escape zone
        self.seeker_vertical_half = np.radians(60)
        self.seeker_horizontal_half = np.radians(60) 

        #Missile configs (aim9x)
        sound_mps = 300.0 #speed of sound (m/s, mach) universal number for 10k - 30k ft
        self.missile_speed_max = 3.0 * sound_mps #max speed m/s
        self.max_flight_time = 30.0 #second
        self.lethal_radius = 7.0    #meters
        self.missile_count = 1   
        self.missile_max_g = 40.0 

    def reset(self, seed=None, options = None): #IMPORTANT: make sure to reset any CONSUMABLE units, trims maybe in the future
        super().reset(seed=seed)
        self.fdm['ic/h-sl-ft'] = 20000.0#self.np_random.integers(8000, 12000) #randomize the starting position of the aircraft
        self.fdm['ic/vc-kts'] = 450.0 #self.np_random.integers(350,400)  #knots
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
        self.prev_rudder = 0.0
        self.prev_throttle = 0.0
        self.elev_cmd = 0.0
        self.aile_cmd = 0.0
        self.rud_cmd = 0.0

        #bandit stats
        self.lat_agent = self.fdm['position/lat-geod-deg']
        self.lon_agent = self.fdm['position/long-gc-deg'] 
        self.bandit_vel = np.array([127.3, 127.3, 0.0])   #due north, not changing altitude, 0 vertical speed
                                    #north,east,up(vs)
        self.bandit_pos = np.array([4774.0, 4774.0, 9144.0]) #due north, at 30000ft
                                    #north,east,up(alt)
        
        self.prev_heading = self.fdm['attitude/psi-rad']
        self.turned = 0.0   #accumulator
        self.prev_pitch_rate = 0.0
        obs = self._get_obs()   #contains the 8 observation data from def _get_obs
        self.prev_range_err = self.range_err()
        self.prev_off_angle = self.off_angle

        info = {}
        return obs, info
    
    def _get_obs(self):

        relative_data = self.bandit_pos - self.agent_pos()
        range = np.linalg.norm(relative_data)
        los_hat = relative_data / (range + 1e-9) #normalize range, leaving the pure direction 
        #3D cone
        pitch_angle = self.fdm['attitude/theta-rad']
        heading_angle = self.fdm['attitude/psi-rad']  #from north's perspective
        nose_vec = np.array([np.cos(pitch_angle) * np.cos(heading_angle),   #North
                             np.cos(pitch_angle) * np.sin(heading_angle),   #East
                             np.sin(pitch_angle)])                          #Up
        self.range = float(range)
        bearing = np.arctan2(relative_data[1], relative_data[0]) #Absolute bearing: from north 
        angle_off = (bearing - self.fdm['attitude/psi-rad'] + np.pi) % (2 * np.pi) - np.pi     #Relative bearing: from agent's nose
        self.off_angle = float(np.arccos(np.clip(np.dot(nose_vec, los_hat), -1.0, 1.0)))
        relative_alt = relative_data[2]
        agent_vel = np.array([self.fdm['velocities/v-north-fps'] * 0.3048,
                              self.fdm['velocities/v-east-fps'] * 0.3048,
                              -self.fdm['velocities/v-down-fps'] * 0.3048])
        closure = -np.dot(self.bandit_vel - agent_vel, relative_data/(range+1e-9)) #gap shrinking / expanding rate
        
        bandit_state = np.array([range, angle_off, relative_alt, closure], dtype=np.float32)
        agent_state = np.array(
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
            self.prev_elev,
            self.prev_aile,
            self.prev_rudder,
            self.prev_throttle,    
            ], dtype = np.float32
        )

        return np.concatenate([agent_state, bandit_state])
    def agent_pos(self):
        lat = self.fdm['position/lat-geod-deg']
        lon = self.fdm['position/long-gc-deg']
        alt = self.fdm['position/h-sl-meters']

        #three components of the velocity
        north = (lat - self.lat_agent) * 111320.0 #deg lat -> meters
        east = (lon - self.lon_agent) * 111320.0 * np.cos(np.radians(self.lat_agent))
        up = alt

        return np.array([north, east, up])        
    
    def range_err (self):
        return max (0.0, self.range - self.rmax) + max(0.0, self.rmin - self.range)

    def step(self, action):
        a = 0.4
        self.elev_cmd = a * float(action[1]) + (1-a) * self.elev_cmd
        self.aile_cmd = a * float(action[2]) + (1-a) * self.aile_cmd
        self.rud_cmd = a * float(action[2]) + (1-a) * self.rud_cmd

        self.fdm['fcs/throttle-cmd-norm'] = float ((action[0] + 1.0) / 2.0)   #assign value back to the self.action_space
        self.fdm['fcs/elevator-cmd-norm'] = self.elev_cmd
        self.fdm['fcs/aileron-cmd-norm'] = self.aile_cmd
        self.fdm['fcs/rudder-cmd-norm'] = self.rud_cmd
        self.fdm["gear/gear-cmd-norm"] = 0.0
        #run 
        for _ in range(self.sim_steps_per_action):
            self.fdm.run()
        dt = self.fdm.get_delta_t() * self.sim_steps_per_action #sync the bandit with agent, 0.1s per update
        self.bandit_pos += self.bandit_vel * dt
        
        #get obs
        obs = self._get_obs()

        self.curr_step += 1
        alt_agl_m = self.fdm['position/h-sl-ft'] * 0.3048
        crashed = bool((alt_agl_m < 30) or abs(self.fdm['attitude/phi-rad']) > np.radians(100))
        terminated = crashed
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
        #air speed policy  
        if speed_knots < 350:
            reward -= 0.05 * abs(speed_knots - 350)
        #elif speed_knots > 400:
        #    reward -= 0.05 * (speed_knots - 400)

        #Elevator anti bang bang
        delta_elev = abs(self.elev_cmd - self.prev_elev)
        reward -= 0.2 * delta_elev
        self.prev_elev = self.elev_cmd

        #Aileron anti bang bang
        delta_aile = abs(self.aile_cmd - self.prev_aile)
        reward -= 0.2 * delta_aile
        self.prev_aile = self.aile_cmd
        
        reward -= 0.6 * (abs(self.elev_cmd) + abs(self.aile_cmd) + abs(self.rud_cmd))
        #anti bang bang
        reward -= 0.1 * abs(roll_rate)
        d_pitch_rate = abs(pitch_rate - self.prev_pitch_rate)
        reward -= 0.2 * d_pitch_rate
        self.prev_pitch_rate = pitch_rate
        self.prev_throttle = action[0]
        self.prev_rudder = self.rud_cmd
        
        #maneuver policy: shorten range and position seeker cone
        #range
        range_error = self.range_err()
        reward += 0.02 * (self.prev_range_err - range_error)
        self.prev_range_err = range_error
        if range_error == 0.0: 
            reward += 0.5
        #seeker cone
        angle_diff = self.off_angle
        reward += 3.0 * (self.prev_off_angle - angle_diff)
        self.prev_off_angle = angle_diff
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