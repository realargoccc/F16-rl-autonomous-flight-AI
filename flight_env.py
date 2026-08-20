import gymnasium as gym
import pickle
import numpy as np
import jsbsim
import os
import random
import math
from collections import namedtuple
from gymnasium.spaces import Box
from stable_baselines3 import PPO

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
        self.thr_cmd = 0.0
        self.reset_obs_memory()

    def reset_obs_memory(self):
        self.prev_elev = 0.0
        self.prev_aile = 0.0
        self.prev_rudder = 0.0
        self.prev_throttle = 0.0
        self.prev_obs_boresight_az = None
        self.prev_obs_boresight = None

    def __getitem__(self, k): #self.me for bandit, self.fdm for agent
        return self.fdm[k]
    
    def __setitem__(self, k, v):
        self.fdm[k] = v

    def run_ic(self):
        self.fdm.run_ic()
        self.elev_cmd = self.aile_cmd = self.rudd_cmd = 0.0
        self.thr_cmd = 0.5
        self.reset_obs_memory()

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

    def ctrl_input(self, action, thr_rate=0.1, rate=0.5):
        self.elev_cmd += np.clip(float(action[1]) - self.elev_cmd, -rate, rate)
        self.aile_cmd += np.clip(float(action[2]) - self.aile_cmd, -rate, rate)
        self.rudd_cmd += np.clip(float(action[3]) - self.rudd_cmd, -rate, rate)
        self.thr_cmd  += np.clip(float((action[0] + 1.0) / 2.0) - self.thr_cmd, -thr_rate, thr_rate)

        self.fdm['fcs/throttle-cmd-norm'] = float((action[0] + 1.0) / 2.0)
        self.fdm['fcs/elevator-cmd-norm'] = self.elev_cmd
        self.fdm['fcs/aileron-cmd-norm'] = self.aile_cmd
        self.fdm['fcs/rudder-cmd-norm'] = 0.0
        self.fdm['gear/gear-cmd-norm'] = 0.0

    def maneuver(self, exp_heading, exp_pitch, exp_speed): #k = conversion rate
        k_hdg, k_bank, k_aile = 2.0, 1.5, 0.5
        k_vs, k_elev = 0.03, 1.5
        k_speed, thrt_bias = 0.02, 0.5
        max_bank = np.radians(75.0)

        #heading: -pi to pi
        heading_err = (exp_heading - self['attitude/psi-rad'] + np.pi) % (2*np.pi) - np.pi
        exp_bank = np.clip(k_hdg * heading_err, -max_bank, max_bank)
        bank_norm = float(np.clip(exp_bank / np.radians(80.0), -1.0, 1.0))

        #bank
        exp_bank = np.clip(k_hdg * heading_err, -max_bank, max_bank)
        bank_err = exp_bank - self['attitude/phi-rad']
        roll_rate = self['velocities/p-rad_sec']
        aile = np.clip(k_bank * bank_err - k_aile * roll_rate, -1.0, 1.0)

        #pitch
        v_true = self['velocities/vt-fps'] * 0.3048
        exp_vs = v_true * np.sin(exp_pitch)
        vs = self['velocities/h-dot-fps'] * 0.3048
        pitch_rate = self['velocities/q-rad_sec']
        elev = np.clip(-(k_vs * (exp_vs - vs) - k_elev * pitch_rate), -1.0, 1.0)

        #speed
        spd_err = exp_speed - self['velocities/vt-fps'] * 0.3048
        throttle = np.clip(thrt_bias + k_speed * spd_err, 0.0, 1.0) * 2.0 - 1.0

        return np.array([throttle, elev, aile, 0.0], dtype=np.float32)

ObsState = namedtuple("ObsState", ["range", "boresight", "boresight_az", "closure",
                                   "omega_yaw", "omega_pitch", "aspect_angle"])


class F16Env(gym.Env):
    def __init__(self):
        self.me = Aircraft()
        self.foe = Aircraft()
        super().__init__()
        self.observation_space = Box(low=-np.inf, high = np.inf, shape=(30,), dtype = np.float32)    #set throttle and elevator lower and upper bound
        self.action_space = Box(low = np.array([-1.0, -1.0, -1.0, -1.0], dtype = np.float32),
                                high = np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32), dtype = np.float32)
        self.max_episodes_steps = 600
        self.curr_step = 0
        self.target_alt_ft = 10000.0
        self.sim_steps_per_action = 12

        #WEZ (Weapon Engagement Zone) configs
        self.max_hp = 1.0
        self.gun_rmin = 450.0
        self.gun_rmax = 900.0
        self.gun_cone = np.radians(3.0)
        self.k_damage = 60.0 # 2 reward per 0.1 hp damage dealt
        self.mirror_obs = np.array([6, 7, 11, 12, 14, 19, 24, 26])
        self.range_band = (700.0, 1200.0)
        self.aspect_band = (0.0, 80.0)
        self.climb_deg = 15.0
        self.dive_deg = 8.0
        self.aim_width = np.radians(20.0)
        self.defensive_p = 0.0
        #opponent pool
        self.foe_pool = []
        self.foe_pool_prob = 0.5
        self.foe_policy = None

    @property
    def range(self):        return self.me_state.range
    @property
    def boresight(self):    return self.me_state.boresight
    @property
    def boresight_az(self): return self.me_state.boresight_az
    @property
    def closure(self):      return self.me_state.closure
    @property
    def omega_yaw(self):    return self.me_state.omega_yaw
    @property
    def omega_pitch(self):  return self.me_state.omega_pitch
    @property
    def aspect_angle(self): return self.me_state.aspect_angle

    def load_foe(self, tag):
        model = PPO.load("ppo_f16_eleva_" + tag + ".zip", device = "cpu")
        with open("vecnorm_eleva_" + tag + ".pkl", "rb") as fh:
            vn = pickle.load(fh)
        self.foe_pool.append((model, vn.obs_rms, float(vn.clip_obs), float(vn.epsilon)))
        return len(self.foe_pool)


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
        #bool(self.np_random.random() < 0.5)
        self.mirror = False  
        #foe's tactic:
        if self.np_random.random() < 0.35: #run away 
            self.turn_offset = np.pi
        else:
            self.turn_offset = float(self.np_random.choice([-1.0, 1.0])) * np.pi/2 #beam 三九机动

        sign = float(self.np_random.choice([-1.0, 0.0, 1.0]))
        mag = self.dive_deg if sign < 0 else self.climb_deg
        self.pitch_target = sign * np.radians(mag) #descend, level, climb

        #foe spawn:
        self.setup = "offensive"
        foe_range = float(self.np_random.uniform(*self.range_band))
        aspect_sign = float(self.np_random.choice([-1.0, 1.0]))

        self.spawn_aspect = aspect_sign * float(self.np_random.uniform(*self.aspect_band))
        foe_heading = self.spawn_aspect % 360.0
        foe_east = 0.0
        self.nominal_speed = 300.0
        if len(self.foe_pool) > 0 and self.np_random.random() < self.foe_pool_prob:
            self.foe_policy = self.foe_pool[(self.np_random.integers(len(self.foe_pool)))]
        else:
            self.foe_policy = None

        #Foe spawn configs
        foe_spawn_low, foe_spawn_high = self.np_random.choice([(-500.0, -250.0), (250.0, 500.0)])
        #shuffle defense and offense spawn
        side = -1.0 if self.np_random.random() < self.defensive_p else 1.0
        self.setup = "defensive" if side < 0 else "offensive" 
        
        foe_rel_alt = self.np_random.uniform(foe_spawn_low, foe_spawn_high)
        self.foe['ic/lat-gc-deg'] = lat0 + foe_range / 111320.0
        self.foe['ic/long-gc-deg'] = lon0 + foe_east / (111320.0 * np.cos(np.radians(lat0)))
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
        obs, self.me_state = self._get_obs(self.me, self.foe, self.agent_hp, self.foe_hp)  #contains the 8 observation data from def _get_obs
        self.foe_obs, self.foe_state = self._get_obs(self.foe, self.me, self.foe_hp, self.agent_hp)
        #delete this self.prev_range_err = self.range_err()
        info = {}
        return obs, info
        
    def _get_obs(self, me, foe, own_hp, foe_hp):
        foe_pos = foe.pos()
        foe_vel = foe.vel()
        relative_data = foe_pos - me.pos()
        range = np.linalg.norm(relative_data)
        los_hat = relative_data / (range + 1e-9) #normalize range, leaving the pure direction 
        
        #3D cone
        pitch_angle = me['attitude/theta-rad']
        heading_angle = me['attitude/psi-rad']  #from north's perspective
        nose_vec = np.array([np.cos(pitch_angle) * np.cos(heading_angle),   #North
                             np.cos(pitch_angle) * np.sin(heading_angle),   #East
                             np.sin(pitch_angle)])                          #Up
        bearing = np.arctan2(relative_data[1], relative_data[0]) #Absolute bearing: from north 
        boresight_az = (bearing - me['attitude/psi-rad'] + np.pi) % (2 * np.pi) - np.pi     #Relative bearing: from agent's nose
        boresight = float(np.arccos(np.clip(np.dot(nose_vec, los_hat), -1.0, 1.0)))
        relative_alt = relative_data[2]
        agent_vel = me.vel()
        closure = -np.dot(foe_vel - agent_vel, relative_data/(range+1e-9)) #gap shrinking / expanding rate

        #LOS roration rate - lead pursuit logging
        phi = me['attitude/phi-rad']
        right0 = np.array([-np.sin(heading_angle), np.cos(heading_angle), 0.0])
        up0 = np.cross(nose_vec, right0)
        body_right = right0 * np.cos(phi) - up0 * np.sin(phi)
        body_up = right0 * np.sin(phi) + up0 * np.cos(phi)

        #LOS rotation rate vector
        rel_vel = foe_vel - agent_vel
        omega = np.cross(relative_data, rel_vel) / (range**2 + 1e-9)

        omega_scale = np.radians(30.0)
        omega_yaw = float(np.dot(omega, body_up)) / omega_scale
        omega_pitch = float(np.dot(omega, body_right)) / omega_scale

        #boresight error rates
        if me.prev_obs_boresight_az is None:
            me.prev_obs_boresight_az = boresight_az
            me.prev_obs_boresight = boresight

        dt_obs = me.get_delta_t() * self.sim_steps_per_action
        rate_scale = dt_obs * np.radians(30.0)
        d_boresight_az = (boresight_az - me.prev_obs_boresight_az + np.pi) % (2*np.pi) - np.pi

        boresight_az_rate = d_boresight_az / rate_scale
        boresight_rate = (boresight - me.prev_obs_boresight) / rate_scale

        me.prev_obs_boresight_az = float(boresight_az)
        me.prev_obs_boresight = float(boresight)

        foe_speed = float(np.linalg.norm(foe_vel)) + 1e-9
        aspect_ang = float(np.arccos(np.clip(np.dot(foe_vel / foe_speed, -relative_data / (range + 1e-9)), -1.0, 1.0)))
        aspecta_norm = (aspect_ang - np.pi / 2) / (np.pi / 2)
        foe_bs = foe.boresight_to(me.pos())
        bandit_state = np.array([range, boresight_az, relative_alt, closure, foe_hp, boresight, boresight_az_rate, boresight_rate,
                                 omega_yaw, omega_pitch, aspecta_norm, foe_bs], dtype=np.float32)
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
            me.prev_elev,
            me.prev_aile,
            me.prev_rudder,
            me.prev_throttle,
            own_hp,
            ], dtype = np.float32
        )
        obs = np.concatenate([agent_state, bandit_state])
        if self.mirror:
            obs[self.mirror_obs] *= -1.0

        state = ObsState(float(range), boresight, float(boresight_az), float(closure), omega_yaw,
                         omega_pitch, aspect_ang)
        return obs, state

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if self.mirror:
            action[2] *= -1.0
            action[3] *= -1.0
        if self.foe_policy is None:
            los = self.me.pos() - self.foe.pos()
            exp_heading = np.arctan2(los[1], los[0]) + self.turn_offset
            foe_action = self.foe.maneuver(exp_heading, self.pitch_target, self.nominal_speed)
        else:
            model, rms, clip, eps = self.foe_policy #rms = runningmeanstd
            nobs = np.clip((self.foe_obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip)
            foe_action, _ = model.predict(nobs.astype(np.float32), deterministic=True)
        self.foe.ctrl_input(foe_action)
        self.me.ctrl_input(action)

        #run 
        self.me.run(self.sim_steps_per_action)
        self.foe.run(self.sim_steps_per_action)
        dt = self.me.get_delta_t() * self.sim_steps_per_action #sync the bandit with agent, 0.1s per update
        
        obs, self.me_state = self._get_obs(self.me, self.foe, self.agent_hp, self.foe_hp)
        self.foe_obs, self.foe_state = self._get_obs(self.foe, self.me, self.foe_hp, self.agent_hp)

        self.curr_step += 1
        alt_agl_m = self.me['position/h-agl-ft'] * 0.3048
        truncated = bool(self.curr_step >= self.max_episodes_steps)
        speed_knots = self.me['velocities/vc-fps'] * 0.592484    #speed in knots
        curr_throttle = self.me['fcs/throttle-cmd-norm']
        #Turning Policy Units
        curr_heading = self.me['attitude/psi-rad'] 
        foe_alt_agl_m = self.foe['position/h-agl-ft'] * 0.3048
        foe_crashed = bool(foe_alt_agl_m < 30) or abs(self.foe['accelerations/Nz']) > 13.0
        curr_bank = self.me['attitude/phi-deg'] 
        curr_g = self.me['accelerations/Nz']
        aim_cone = np.radians(25.0)
        crashed = bool(alt_agl_m < 30) or abs(self.me['accelerations/Nz']) > 13.0 or abs(curr_g) > 13.0

        delta_turn = (curr_heading - self.prev_heading + np.pi) % (2*np.pi) - np.pi
        self.turned += delta_turn
        self.prev_heading = curr_heading

        reward = 0.0
        # constraint rails — flat interior, wall at the edge
        if speed_knots < 350:
            reward -= 0.01 * (350 - speed_knots)
        elif speed_knots > 800:
            reward -= 0.01 * (speed_knots - 800)
        if curr_g > 8.5:
            reward -= 0.5 * (curr_g - 8.5)**2       #g back-off ramp
        elif curr_g < -1.0:
            reward -= 0.5 * (-1.0 - curr_g)**2

        #below deck punishment
        alt_agl_kft = alt_agl_m / 304.8
        if alt_agl_kft < 3.0:
            reward -= 0.5 * (3.0 - alt_agl_kft) ** 2

        #punish huge oscillation 
        a_t = np.asarray(action[0:4], dtype = np.float32)
        a_t1 = self.prev_action[0:4]
        a_t2 = self.prev_prev_action[0:4]

        reward -= 0.02 * float(np.sum((a_t - 2.0 * a_t1 + a_t2) ** 2))  #curvature punishment (二阶差)

        #wez agent's configs
        in_wez = (self.gun_rmin <= self.range <= self.gun_rmax)
        if in_wez:
            pk = math.exp(-(self.boresight / self.gun_cone) ** 2)
            damage = dt * (self.gun_rmin / self.range) * pk
            self.foe_hp -= damage
            reward += self.k_damage * damage
        #wez bandit's configs
        foe_boresight = self.foe.boresight_to(self.me.pos())
        if (foe_boresight < self.gun_cone) and (self.gun_rmin <= self.range <= self.gun_rmax):
            damage = dt * (self.gun_rmin / self.range)
            self.agent_hp -= damage
            reward -= self.k_damage * damage

        #distance away
        dis = max(0.0, self.range - self.gun_rmax) + max(0.0, self.gun_rmin - self.range)
        reward -= 1.0 * min(dis / 1000.0, 1.0)

        #symmetric pair
        aim = math.exp(-(self.boresight / self.aim_width) ** 2)
        threat = math.exp(-(foe_boresight / self.aim_width) ** 2)
        reward += 0.5 * aim
        reward -= 0.5 * threat

        win = bool(self.foe_hp <= 0.0)
        lose = bool(self.agent_hp <= 0) #knock it off - fights over
        if crashed: reward -= 300
        if win: reward += 400.0
        if lose: reward -= 400.0
        terminated = crashed or lose or win or foe_crashed

        # foe and agent bookkeeping — feeds the observation
        self.me.prev_elev, self.me.prev_aile = self.me.elev_cmd, self.me.aile_cmd
        self.me.prev_rudder, self.me.prev_throttle = self.me.rudd_cmd, action[0]
        self.foe.prev_elev, self.foe.prev_aile = self.foe.elev_cmd, self.foe.aile_cmd
        self.foe.prev_rudder, self.foe.prev_throttle = self.foe.rudd_cmd, foe_action[0]
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = np.array(action, dtype=np.float32)


        info = {"crashed": crashed, "foe_crashed": foe_crashed, "win":win}
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