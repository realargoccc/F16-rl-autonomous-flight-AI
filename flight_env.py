import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import jsbsim
import os
import random
import math

ROOT = os.path.join(os.path.dirname(__file__), "jsbsim-data")

class Aircraft():
    def __init__(self):
        self.fdm = jsbsim.FGFDMExec(ROOT, None)
        self.fdm.set_debug_level(0)
        self.fdm.load_model('f16')
        self.lat0 = 0.0
        self.lon0 = 0.0
        self.elev_cmd = 0.0
        self.aile_cmd = 0.0
        self.rudd_cmd = 0.0

    def __getitem__(self, k): #self.me for bandit, self.fdm for agent
        return self.fdm[k]
    
    def __setitem__(self, k, v):
        self.fdm[k] = v

    def run_ic(self):
        self.fdm.run_ic()
        self.elev_cmd = self.aile_cmd = self.rudd_cmd = 0.0

    def get_delta_t(self): 
        return self.fdm.get_delta_t()

    def get_sim_time(self):
        return self.fdm.get_sim_time()

    def run(self, n):
        for _ in range(n):
            self.fdm.run()

    def set_origin(self, lat, lon): #both share the same fight location
        self.lat0, self.lon0 = lat, lon

    def pos(self):
        lat = self.fdm['position/lat-geod-deg']
        lon = self.fdm['position/long-gc-deg']
        alt = self.fdm['position/h-sl-meters']
        north = (lat - self.lat0) * 111320.0
        east = (lon - self.lon0) * 111320.0 * np.cos(np.radians(self.lat0))
        return np.array([north, east, alt])

    #for boresight_to angle calculation
    def nose(self): 
        pitch = self.fdm['attitude/theta-rad']
        heading = self.fdm['attitude/psi-rad']

        return np.array([np.cos(pitch) * np.cos(heading),
                         np.cos(pitch) * np.sin(heading),
                         np.sin(pitch)])

    def boresight_to(self, target_pos):
        los = target_pos - self.pos()
        los_hat = los / (np.linalg.norm(los) + 1e-9)

        return float(np.arccos(np.clip(np.dot(self.nose(), los_hat), -1.0, 1.0)))
    
    def vel(self): #NEU frame 
        return np.array([self.fdm['velocities/v-north-fps'] * 0.3048,
                         self.fdm['velocities/v-east-fps'] * 0.3048,
                        -self.fdm['velocities/v-down-fps'] * 0.3048])

    def ctrl_input(self, action):
        smooth = 0.7
        self.elev_cmd = smooth * float(action[1]) + (1 - smooth) * self.elev_cmd
        self.aile_cmd = smooth * float(action[2]) + (1 - smooth) * self.aile_cmd
        self.rudd_cmd = smooth * float(action[3]) + (1 - smooth) * self.rudd_cmd

        self.fdm['fcs/throttle-cmd-norm'] = float((action[0] + 1.0) / 2.0)
        self.fdm['fcs/elevator-cmd-norm'] = self.elev_cmd
        self.fdm['fcs/aileron-cmd-norm'] = self.aile_cmd
        self.fdm['fcs/rudder-cmd-norm'] = self.rudd_cmd
        self.fdm['gear/gear-cmd-norm'] = 0.0

    def maneuver(self, exp_heading, exp_pitch, exp_speed): #k = conversion rate
        k_hdg, k_bank, k_aile = 2.0, 1.5, 0.5
        k_vs, k_elev = 3.0, 2.0
        k_speed, thrt_bias = 0.02, 0.5
        max_bank = np.radians(75.0)

        #heading: -pi to pi
        heading_err = (exp_heading - self['attitude/psi-rad'] + np.pi) % (2*np.pi) - np.pi
        exp_bank = np.clip(k_hdg * heading_err, -max_bank, max_bank)

        #bank
        bank_err = exp_bank - self['attitude/phi-rad']
        roll_rate = self['velocities/p-rad_sec']
        aile = np.clip(k_bank * bank_err - k_aile * roll_rate , -1.0, 1.0)

        #pitch
        exp_vs = exp_speed * np.sin(exp_pitch)
        vs = self['velocities/h-dot-fps'] * 0.3048
        pitch_rate = self['velocities/q-rad_sec']
        elev = np.clip(-(k_vs * (exp_vs - vs) - k_elev * pitch_rate), -1.0, 1.0)

        #speed
        spd_err = exp_speed - self['velocities/vt-fps'] * 0.3048
        throttle = np.clip(thrt_bias + k_speed * spd_err, 0.0, 1.0) * 2.0 - 1.0

        return np.array([throttle, elev, aile, 0.0], dtype=np.float32)

class F16Env(gym.Env):
    def __init__(self):
        self.me = Aircraft()
        self.foe = Aircraft()
        super().__init__()
        self.observation_space = Box(low=-np.inf, high = np.inf, shape=(26,), dtype = np.float32)    #set throttle and elevator lower and upper bound
        self.action_space = Box(low = np.array([-1.0, -1.0, -1.0, -1.0], dtype = np.float32),
                                high = np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32), dtype = np.float32)
        self.max_episodes_steps = 300
        self.curr_step = 0
        self.target_alt_ft = 10000.0
        self.sim_steps_per_action = 12

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
        #agent data
        self.me['ic/h-sl-ft'] = self.np_random.integers(18000, 25000) #randomize the starting position of the aircraft
        self.me['ic/vc-kts'] = 450.0 #self.np_random.integers(350,400)  #knots
        self.me['ic/throttle-cmd-norm'] = 0.5
        self.me['ic/elevator-cmd-norm'] = 0.0
        self.me["gear/gear-cmd-norm"] = 0.0
        self.me['propulsion/tank[0]/contents-lbs'] = 1500.0 #set initial fuel tank
        self.me['propulsion/tank[1]/contents-lbs'] = 1500.0
        self.me['propulsion/engine/set-running'] = 1.0      #Make sure the engine starts
        self.me['ic/phi-deg'] = 0.0 #wings level (no bank)
        self.me['ic/psi-true-deg'] = 0.0 # spawn due northing heading 000
        self.me.run_ic()
        #counter reset
        #self.me['simulation/do_simple_trim'] = 1  #one time solution before agent take over
        #bandit data
       

        self.curr_step = 0
        self.prev_elev = 0.0
        self.prev_aile = 0.0
        self.prev_rudder = 0.0
        self.prev_throttle = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32) #currently 4 actions in action space
        self.prev_prev_action = np.zeros(4, dtype=np.float32)

        #set fight location for both
        lat0 = self.me['position/lat-geod-deg']
        lon0 = self.me['position/long-gc-deg']
        self.me.set_origin(lat0, lon0)
        self.foe.set_origin(lat0, lon0)

        self.prev_heading = self.me['attitude/psi-rad']
        self.turned = 0.0   #accumulator
        self.prev_pitch_rate = 0.0
        #foe's tactic:
        if self.np_random.random() < 0.35: #run away 
            self.turn_offset = np.pi
        else:
            self.turn_offset = float(self.np_random.choice([-1.0, 1.0])) * np.pi/2 #beam 三九机动
        self.pitch_target = float(self.np_random.choice([-1.0, 0.0, 1.0])) * np.radians(15.0) #descend, level, climb

        #foe spawn:
        if self.np_random.random() < 0.2: #20% heads on
            self.setup = "head_on"
            foe_range = 1800.0
            foe_heading = 180.0
        else:
            self.setup = "offensive"
            foe_range = 700.0
            foe_heading = 0.0

        self.nominal_speed = 300.0
        #Foe spawn configs

        foe_spawn_low, foe_spawn_high = self.np_random.choice([(-500.0, -250.0), (250.0, 500.0)]) 
        foe_rel_alt = self.np_random.uniform(foe_spawn_low, foe_spawn_high)
        self.foe['ic/lat-gc-deg'] = lat0 + foe_range / 111320.0
        self.foe['ic/long-gc-deg'] = lon0
        self.foe['ic/h-sl-ft'] = (self.me['position/h-sl-meters'] + foe_rel_alt) / 0.3048 #agent's perspective 
        self.foe['ic/vc-kts'] = 450.0
        self.foe['ic/throttle-cmd-norm'] = 0.5
        self.foe['propulsion/tank[0]/contents-lbs'] = 1500.0
        self.foe['propulsion/tank[1]/contents-lbs'] = 1500.0
        self.foe['propulsion/engine/set-running'] = 1.0
        self.foe['ic/phi-deg'] = 0.0
        self.foe['ic/psi-true-deg'] = foe_heading
        self.foe.run_ic()

        self.foe_hp = 1.0
        self.agent_hp = 1.0
        self.prev_obs_boresight_az = None
        self.prev_obs_boresight = None
        obs = self._get_obs(self.me, self.foe.pos(), self.foe.vel(), self.foe_hp)  #contains the 8 observation data from def _get_obs
        #delete this self.prev_range_err = self.range_err()
        self.prev_boresight = self.boresight
        self.prev_gap = max(0.0, self.range - self.gun_rmax) + max(0.0, self.gun_rmin - self.range)
        info = {}
        return obs, info
        
    def _get_obs(self, me, foe_pos, foe_vel, foe_hp):
        relative_data = foe_pos - me.pos()
        range = np.linalg.norm(relative_data)
        los_hat = relative_data / (range + 1e-9) #normalize range, leaving the pure direction 
        
        #3D cone
        pitch_angle = me['attitude/theta-rad']
        heading_angle = me['attitude/psi-rad']  #from north's perspective
        nose_vec = np.array([np.cos(pitch_angle) * np.cos(heading_angle),   #North
                             np.cos(pitch_angle) * np.sin(heading_angle),   #East
                             np.sin(pitch_angle)])                          #Up
        self.range = float(range) 
        bearing = np.arctan2(relative_data[1], relative_data[0]) #Absolute bearing: from north 
        boresight_az = (bearing - me['attitude/psi-rad'] + np.pi) % (2 * np.pi) - np.pi     #Relative bearing: from agent's nose
        self.boresight = float(np.arccos(np.clip(np.dot(nose_vec, los_hat), -1.0, 1.0)))
        relative_alt = relative_data[2]
        agent_vel = np.array([me['velocities/v-north-fps'] * 0.3048,
                              me['velocities/v-east-fps'] * 0.3048,
                              -me['velocities/v-down-fps'] * 0.3048])
        closure = -np.dot(foe_vel - agent_vel, relative_data/(range+1e-9)) #gap shrinking / expanding rate
        self.closure = float(closure)
        self.boresight_az = float(boresight_az) #for eval logging

        #LOS rate - lead pursuit logging
        if self.prev_obs_boresight_az is None: # if no angle are observed by agent
            self.prev_obs_boresight_az = boresight_az 
            self.prev_obs_boresight = self.boresight
        
        dt_obs = me.get_delta_t() * self.sim_steps_per_action
        scale = dt_obs * np.radians(30.0)
        d_boresight_az = (boresight_az - self.prev_obs_boresight_az + np.pi) % (2*np.pi) - np.pi

        boresight_az_rate = d_boresight_az / scale
        boresight_rate = (self.boresight - self.prev_obs_boresight) / scale

        self.prev_obs_boresight_az = float(boresight_az)
        self.prev_obs_boresight    = float(self.boresight)

        rel_vel = (foe_vel - agent_vel) / 300
        bandit_state = np.array([range, boresight_az, relative_alt, closure, foe_hp, self.boresight, 
                                 boresight_az_rate, boresight_rate], dtype=np.float32)
        agent_state = np.array(
            [me['position/h-sl-meters'],          #altitude
            me['velocities/vc-fps'] * 0.3048,     #IAS
            me['attitude/theta-rad'],             #pitch
            me['velocities/q-rad_sec'],           #pitch rate
            me['velocities/h-dot-fps'] * 0.3048,  #vertical speed
            me['aero/alpha-deg'],                 #aoa-deg
            me['attitude/phi-rad'],               #bank angle in radians
            me['velocities/p-rad_sec'],           #roll rate
            me['propulsion/engine/n1'],           #engine rpm (low lag responder to throttle)
            me['accelerations/Nz'],               #g_load
            me['velocities/mach'],                #corner speed monitor
            me['velocities/r-rad_sec'],           #yaw rate
            me['aero/beta-deg'],                  #sideslip (yaw angle)
            self.prev_elev,
            self.prev_aile,
            self.prev_rudder,
            self.prev_throttle,
            self.agent_hp,    
            ], dtype = np.float32
        )

        return np.concatenate([agent_state, bandit_state])

    def step(self, action):
        los = self.me.pos() - self.foe.pos()
        exp_heading = np.arctan2(los[1], los[0]) + self.turn_offset
        foe_action = self.foe.maneuver(exp_heading, self.pitch_target, self.nominal_speed)
        self.foe.ctrl_input(foe_action)
        self.me.ctrl_input(action)
        #run 
        
        self.me.run(self.sim_steps_per_action)
        self.foe.run(self.sim_steps_per_action)
        dt = self.me.get_delta_t() * self.sim_steps_per_action #sync the bandit with agent, 0.1s per update
        
        obs = self._get_obs(self.me, self.foe.pos(), self.foe.vel(), self.foe_hp)

        self.curr_step += 1
        alt_agl_m = self.me['position/h-agl-ft'] * 0.3048
        truncated = bool(self.curr_step >= self.max_episodes_steps)
        speed_knots = self.me['velocities/vc-fps'] * 0.592484    #speed in knots
        curr_throttle = self.me['fcs/throttle-cmd-norm']
        #Turning Policy Units
        curr_heading = self.me['attitude/psi-rad'] 
        curr_bank = self.me['attitude/phi-deg'] 
        curr_g = self.me['accelerations/Nz']
        aim_cone = np.radians(25.0)
        crashed = bool(alt_agl_m < 30) or abs(self.me['accelerations/Nz']) > 13.0 or abs(curr_g) > 13.0

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
            reward -= 0.5 * (curr_g - 8.5)**2       #g back-off ramp
        elif curr_g < -1.0:
            reward -= 0.5 * (-1.0 - curr_g)**2

        #punish huge oscillation 
        a_t = np.asarray(action[1:4], dtype = np.float32)
        a_t1 = self.prev_action[1:4]
        a_t2 = self.prev_prev_action[1:4]

        reward -= 0.02 * float(np.sum((a_t - a_t1) ** 2))   #rate punishement (一阶差)
        reward -= 0.15 * float(np.sum((a_t - 2.0 * a_t1 + a_t2) ** 2))  #curvature punishment (二阶差)
        reward -= 0.01 * float(np.sum(a_t ** 2))                       #magnitude 

        #wez agent's configs
        in_wez = (self.boresight < self.gun_cone and self.gun_rmin <= self.range <= self.gun_rmax)
        if in_wez:
            damage = dt * (self.gun_rmin / self.range)
            self.foe_hp -= damage
            reward += self.k_damage * damage
        #wez bandit's configs
        foe_boresight = self.foe.boresight_to(self.me.pos())
        if (foe_boresight < self.gun_cone) and (self.gun_rmin <= self.range <= self.gun_rmax):
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
        reward += 3.0 * (self.prev_boresight - self.boresight) # cone gradient — inert dead-ahead, matters off-boresight
        self.prev_boresight = self.boresight
        reward += 0.4 * math.exp(-(self.boresight / aim_cone) ** 2)

        win = bool(self.foe_hp <= 0.0)
        lose = bool(self.agent_hp <= 0) #knock it off - fights over
        if crashed: reward -= 100
        if win: reward += 100.0
        if lose: reward -= 100.0
        terminated = crashed or lose or win

        # bookkeeping — feeds the observation
        self.prev_elev, self.prev_aile = self.me.elev_cmd, self.me.aile_cmd
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = np.array(action, dtype=np.float32)
        self.prev_rudder, self.prev_throttle = self.me.rudd_cmd, action[0]
        
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