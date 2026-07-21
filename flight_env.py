import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import jsbsim
import os
import random
import math

ROOT = os.path.join(os.path.dirname(__file__), "jsbsim-data")

class Bandit:
    def __init__(self):
        self.speed = 300.0
        self.max_turn_rate = np.radians(6.0)
        self.hp = 1.0
    
    def reset(self, np_random, agent_alt_m):
        range_wez = 700.0 #np_random.uniform(700.0, 800.0) 
        bearing = 0.0 #np_random.uniform(-np.radians(5), np.radians(5))
        rand_low, rand_high = np_random.choice([(-500.0, -250.0), (250.0, 500.0)])
        rel_alt = np_random.uniform(rand_low, rand_high)
        self.pos = np.array([range_wez * np.cos(bearing), range_wez * np.sin(bearing), agent_alt_m + rel_alt])
        self.heading = 0.0 #np_random.uniform(-np.pi, np.pi)
        self.vel = self.speed * ( np.array([np.cos(self.heading), np.sin(self.heading), 0.0]))
        self.hp = 1.0

    def step(self, agent_pos, dt):
        los = agent_pos - self.pos
        desire_enga = np.arctan2(los[1], los[0])
        err = (desire_enga - self.heading + np.pi) % (2*np.pi) - np.pi 
        self.heading += np.clip(err, -self.max_turn_rate * dt, self.max_turn_rate * dt)
        self.vel = self.speed * np.array([np.cos(self.heading), 
                                          np.sin(self.heading), 0.0])
        self.pos += self.vel * dt

    def off_angle_to(self, target_pos):
        los = target_pos - self.pos
        los_hat = los / (np.linalg.norm(los) + 1e-9)
        nose = np.array([np.cos(self.heading), np.sin(self.heading), 0.0])
        return float(np.arccos(np.clip(np.dot(nose, los_hat), -1.0, 1.0)))

class F16Env(gym.Env):
    def __init__(self):
        self.fdm = jsbsim.FGFDMExec(ROOT, None) #load FDM
        self.fdm.set_debug_level(0)             #remove banners of aircraft configurations (hundres loc)
        self.fdm.load_model('f16')              #load f16
        super().__init__()
        self.observation_space = Box(low=-np.inf, high = np.inf, shape=(24,), dtype = np.float32)    #set throttle and elevator lower and upper bound
        self.action_space = Box(low = np.array([-1.0, -1.0, -1.0, -1.0], dtype = np.float32),
                                high = np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32), dtype = np.float32)
        self.max_episodes_steps = 300
        self.curr_step = 0
        self.target_alt_ft = 10000.0
        self.sim_steps_per_action = 12

        self.bandit = Bandit()
        #WEZ (Weapon Engagement Zone) configs
        self.max_hp = 1.0
        self.gun_rmin = 450.0
        self.gun_rmax = 900.0
        self.gun_cone = np.radians(3.0)
        self.k_damage = 20.0 # 2 reward per 0.1 hp damage dealt


        '''
        #Missile configs (aim9x)
        sound_mps = 300.0 #speed of sound (m/s, mach) universal number for 10k - 30k ft
        self.missile_speed_max = 3.0 * sound_mps #max speed m/s
        self.max_flight_time = 30.0 #second
        self.lethal_radius = 7.0    #meters
        self.missile_count = 1   
        self.missile_max_g = 40.0 
        '''
    def reset(self, seed=None, options = None): #IMPORTANT: make sure to reset any CONSUMABLE units, trims maybe in the future
        super().reset(seed=seed)
        self.fdm['ic/h-sl-ft'] = self.np_random.integers(18000, 25000) #randomize the starting position of the aircraft
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
        self.prev_action = np.zeros(4, dtype=np.float32) #currently 4 actions in action space
        self.prev_prev_action = np.zeros(4, dtype=np.float32)

        #bandit stats
        self.lat_agent = self.fdm['position/lat-geod-deg']
        self.lon_agent = self.fdm['position/long-gc-deg'] 

        self.prev_heading = self.fdm['attitude/psi-rad']
        self.turned = 0.0   #accumulator
        self.prev_pitch_rate = 0.0
        self.bandit.reset(self.np_random, self.fdm['position/h-sl-meters'])
        self.agent_hp = 1.0
        obs = self._get_obs()   #contains the 8 observation data from def _get_obs
        #delete this self.prev_range_err = self.range_err()
        self.prev_off_angle = self.off_angle
        self.prev_gap = max(0.0, self.range - self.gun_rmax) + max(0.0, self.gun_rmin - self.range)
        info = {}
        return obs, info
    
    def _get_obs(self):
        relative_data = self.bandit.pos - self.agent_pos()
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
        closure = -np.dot(self.bandit.vel - agent_vel, relative_data/(range+1e-9)) #gap shrinking / expanding rate
        self.closure = float(closure)

        bandit_state = np.array([range, angle_off, relative_alt, closure, self.bandit.hp, self.off_angle], dtype=np.float32)
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
            self.agent_hp,    
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

    def step(self, action):
        a = 0.7
        self.elev_cmd = a * float(action[1]) + (1-a) * self.elev_cmd
        self.aile_cmd = a * float(action[2]) + (1-a) * self.aile_cmd
        self.rud_cmd = a * float(action[3]) + (1-a) * self.rud_cmd

        self.fdm['fcs/throttle-cmd-norm'] = float ((action[0] + 1.0) / 2.0)   #assign value back to the self.action_space
        self.fdm['fcs/elevator-cmd-norm'] = self.elev_cmd
        self.fdm['fcs/aileron-cmd-norm'] = self.aile_cmd
        self.fdm['fcs/rudder-cmd-norm'] = self.rud_cmd
        self.fdm["gear/gear-cmd-norm"] = 0.0
        #run 
        
        for _ in range(self.sim_steps_per_action):
            self.fdm.run()
        dt = self.fdm.get_delta_t() * self.sim_steps_per_action #sync the bandit with agent, 0.1s per update

        self.bandit.step(self.agent_pos(), dt)
        
        obs = self._get_obs()

        self.curr_step += 1
        alt_agl_m = self.fdm['position/h-sl-ft'] * 0.3048
        truncated = bool(self.curr_step >= self.max_episodes_steps)
        speed_knots = self.fdm['velocities/vc-fps'] * 0.592484    #speed in knots
        curr_throttle = self.fdm['fcs/throttle-cmd-norm']
        #Turning Policy Units
        curr_heading = self.fdm['attitude/psi-rad'] 
        curr_bank = self.fdm['attitude/phi-deg'] 
        curr_g = self.fdm['accelerations/Nz']
        aim_cone = np.radians(25.0)
        crashed = bool(alt_agl_m < 30) or abs(self.fdm['accelerations/Nz']) > 9.0 or curr_g > 9.0 or curr_g < -3.0

        delta_turn = (curr_heading - self.prev_heading + np.pi) % (2*np.pi) - np.pi
        self.turned += delta_turn
        self.prev_heading = curr_heading

        #reward computations
        reward = -0.1

        # constraint rails — flat interior, wall at the edge
        if speed_knots < 350:
            reward -= 0.01 * (350 - speed_knots)
        elif speed_knots > 800:
            reward -= 0.01 * (speed_knots - 800)
        if curr_g > 8.5:
            reward -= 0.1 * (curr_g - 8.5)                # g back-off ramp
        elif curr_g < -2.5:
            reward -= 0.1 * (-2.5 - curr_g)

        #punish huge oscillation 
        a_t = np.asarray(action[1:4], dtype = np.float32)
        a_t1 = self.prev_action[1:4]
        a_t2 = self.prev_prev_action[1:4]

        reward -= 0.02 * float(np.sum((a_t - a_t1) ** 2))   #rate punishement (一阶差)
        reward -= 0.15 * float(np.sum((a_t - 2.0 * a_t1 + a_t2) ** 2))  #curvature punishment (二阶差)
        reward -= 0.01 * float(np.sum(a_t ** 2))                       #magnitude 

        #wez agent's configs
        in_wez = (self.off_angle < self.gun_cone and self.gun_rmin <= self.range <= self.gun_rmax)
        if in_wez:
            damage = dt * (self.gun_rmin / self.range)
            self.bandit.hp -= damage
            reward += self.k_damage * damage
        #wez bandit's configs
        bandit_offangle = self.bandit.off_angle_to(self.agent_pos())
        if (bandit_offangle < self.gun_cone) and (self.gun_rmin <= self.range <= self.gun_rmax):
            damage = dt * (self.gun_rmin / self.range)
            self.agent_hp -= damage
            reward -= self.k_damage * damage

        #closing gap policy 
        gap = max(0.0, self.range - self.gun_rmax) + max(0.0, self.gun_rmin - self.range)
        reward += 0.1 * (self.prev_gap - gap)
        self.prev_gap = gap

        #closure policy
        if self.range > self.gun_rmax:
            reward += 0.01 * self.closure
        elif self.range < self.gun_rmin:
            reward -= 0.02 * abs(self.closure)

        #closing cone policy 
        reward += 3.0 * (self.prev_off_angle - self.off_angle) # cone gradient — inert dead-ahead, matters off-boresight
        self.prev_off_angle = self.off_angle
        reward += 0.4 * math.exp(-(self.off_angle / aim_cone) ** 2)

        win = bool(self.bandit.hp <= 0.0)
        lose = bool(self.agent_hp <= 0) #knock it off - fights over
        if crashed: reward -= 100
        if win: reward += 100.0
        if lose: reward -= 100.0
        terminated = crashed or lose or win

        # bookkeeping — feeds the observation
        self.prev_elev,   self.prev_aile     = self.elev_cmd, self.aile_cmd
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = np.array(action, dtype=np.float32)
        self.prev_rudder, self.prev_throttle = self.rud_cmd,  action[0]
        
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
            print(f"Episode ended at step {i}, terminated = {terminated}, truncated = {truncated}, total reward: {total_reward:.1f}")
            episode_rewards.append(total_reward)
            total_reward = 0.0
            obs, info = env.reset()

    print(f"Wrapper validation completed, reward:{reward}")
#run command: python flight_env.py
#interpretor select command: /Users/y/Desktop/jsbsim-rl/.venv/bin/pythons
#commit and push command: git add -A, git commit -m "message", git push
#pull from pc: git fetch origin, git reset --hard origin/main