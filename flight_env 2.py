import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import jsbsim
import os

ROOT = os.path.join(os.path.dirname(__file__), "jsbsim-data")

class F16Env(gym.Env):
    def __init__(self):
        self.fdm = jsbsim.FGFDMExec(ROOT, None) #load FDM
        self.fdm.set_debug_level(0)             #remove banners of aircraft configurations (hundres loc)
        self.fdm.load_model('f16')              #load f16
        super().__init__()
        self.observation_space = Box(low=-np.inf, high = np.inf, shape=(6,), dtype = np.float32)    #set throttle and elevator lower and upper bound
        self.action_space = Box(low = np.array([-1.0, -1.0], dtype = np.float32),
                                high = np.array([1.0, 1.0], dtype = np.float32), dtype = np.float32)
        self.max_episodes_steps = 3600
        self.curr_step = 0
        self.target_alt_ft = 10000.0

    def reset(self, seed=None, options = None): #IMPORTANT: make sure to reset any CONSUMABLE units, trims maybe in the future
        super().reset(seed=seed)
        self.fdm['ic/h-sl-ft'] = 10000.0
        self.fdm['ic/vt-fps'] = 425.0  #252 knots
        self.fdm['ic/throttle-cmd-norm'] = 0.5
        self.fdm['ic/elevator-cmd-norm'] = 0.0
        self.fdm["gear/gear-cmd-norm"] = 0.0
        self.fdm['propulsion/tank[0]/contents-lbs'] = 1500.0 #set initial fuel tank
        self.fdm['propulsion/tank[1]/contents-lbs'] = 1500.0
        self.fdm['propulsion/engine/set-running'] = 1.0      #Make sure the engine starts
        self.fdm.run_ic()
        #counter reset
        #self.fdm['simulation/do_simple_trim'] = 1  #one time solution before agent take over
        self.curr_step = 0
        self.prev_elev = 0.0

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
            ], dtype = np.float32
        )
    
    def step(self, action):
        self.fdm['fcs/throttle-cmd-norm'] = float ((action[0] + 1.0) / 2.0)   #assign value back to the self.action_space
        self.fdm['fcs/elevator-cmd-norm'] = float (action[1])
        self.fdm["gear/gear-cmd-norm"] = 0.0
        #run 
        self.fdm.run()      
        #get obs
        obs = self._get_obs()

        self.curr_step += 1
        alt_agl_m = self.fdm['position/h-sl-ft'] * 0.3048
        vertical_speed_rec = self.fdm['velocities/h-dot-fps'] * 0.3048  #unit: m/s
        pitch_rad = self.fdm['attitude/theta-rad']
        terminated = bool(alt_agl_m < 30)
        truncated = bool(self.curr_step >= self.max_episodes_steps)
        curr_alt_ft = self.fdm['position/h-sl-ft']
        target_alt_ft = self.target_alt_ft
        
        #reward computations
        reward = 0.1
        #per step++, reward += 0.1        
        if terminated: #crashed
            reward -= 100.0
        
        #Optimized reward system: initially fix the throttle only train elevator
        alt_diff_ft = curr_alt_ft - target_alt_ft
        reward -= abs(alt_diff_ft) / 1000            #alt diff policy (train alt sense)
        reward -= 0.1 * abs(vertical_speed_rec)      #vs policy (train elevator input)
        reward -= 0.01 * abs(pitch_rad)             #pitch policy (train elevator input)
        #air speed policy  
        speed_knots = self.fdm['velocities/vc-fps'] * 0.592484    #speed in knots
        if speed_knots < 220 or speed_knots > 270:
            reward -= 0.2
        else:
            reward += 0.05
        #Actual flying well bonus
        if abs(alt_diff_ft) < 100 and 220 < speed_knots < 270 and abs(vertical_speed_rec) < 1.0: #
            reward += 0.05
        #Elevator saturation / smoothness limiter
        delta_elev = abs(action[1] - self.prev_elev)
        reward -= 0.05 * delta_elev
        self.prev_elev = action[1]

        #throttle smoothness limiter
        #delta_thrott = abs(action[0] - self.init_thrott)
        #reward -= 0.02 * delta_thrott
        
        
        '''
        #Altitude policy
        elevator_in = self.fdm['fcs/elevator-cmd-norm']
        pitch_deg = self.fdm['attitude/theta-deg']
        #If near target, prefer stable level flight
        if abs(alt_diff_ft) > 0 and abs(alt_diff_ft) < 100:
            reward -= abs(vertical_speed_rec) / 25              #punish wrong VS
            if abs(elevator_in) < 0.05 and abs(pitch_deg) < 2:
                reward += 0.04
            elif abs(elevator_in) >= 0.05 and abs(elevator_in) < 0.1 and abs(pitch_deg) < 4:
                reward += 0.01
            else:
                reward -= 0.05 
        #Below Altitude
        elif alt_diff_ft < 0:  
            reward -= abs(alt_diff_ft) / 10000                  #penalize altitude loss
            if abs(alt_diff_ft) >= 100 and abs(alt_diff_ft) < 500:
                if vertical_speed_rec <= 0: #if below and descend (negative vs)
                    reward -= abs(vertical_speed_rec) / 30          #penalize wrong vertical trend 
                    if vertical_speed_rec > -2.0:    #prefer negative elevator (pull stick)
                        reward += 0.02
                    else:
                        reward -= 0.05                              #punish aggresive neg elev, or any pos eleva
                else: #if below and climbing
                    if vertical_speed_rec < 2.0:
                        reward += 0.04
                    elif vertical_speed_rec >= 2.0 and vertical_speed_rec <=3.0: 
                        reward += 0.01
                    else:
                        reward -= 0.08
            elif abs(alt_diff_ft) >= 500 and abs(alt_diff_ft) < 1000:
                if vertical_speed_rec <= 0: #if below and descend (negative vs)
                    reward -= abs(vertical_speed_rec) / 20          #penalize wrong vertical trend 
                    if vertical_speed_rec > -3.0:    #prefer negative elevator (pull stick)
                        reward += 0.02
                    else:
                        reward -= 0.05                              
                else: #if below and climbing
                    if vertical_speed_rec < 4.0:
                        reward += 0.04
                    elif vertical_speed_rec >= 4.0 and vertical_speed_rec <= 5.0:
                        reward += 0.01
                    else:
                        reward -= 0.08
            else: #abs(alt_diff_ft) >= 1000:
                if vertical_speed_rec <= 0:
                    reward -= abs(vertical_speed_rec) / 15          #penalize wrong vertical trend 
                    if vertical_speed_rec > -4.0:    
                        reward += 0.02
                    else:
                        reward -= 0.05                              
                else: #if below and climbing
                    if vertical_speed_rec < 5.0:
                        reward += 0.04
                    elif vertical_speed_rec >= 5.0 and vertical_speed_rec <= 6.0: 
                        reward += 0.01
                    else:
                        reward -= 0.08
        #Above Altitude
        elif alt_diff_ft > 0:
            reward -= abs(alt_diff_ft) / 10000                  #penalize altitude loss
            if abs(alt_diff_ft) >= 100 and abs(alt_diff_ft) < 500:
                if vertical_speed_rec >= 0: #if Above and climbing: WRONG
                    reward -= abs(vertical_speed_rec) / 30          #penalize wrong vertical trend 
                    if vertical_speed_rec < 2.0:
                        reward += 0.02
                    else:
                        reward -= 0.05                              #punish aggresive neg elev, or any pos eleva
                else:   #if above ane descending: RIGHT
                    if vertical_speed_rec > -2.0:
                        reward += 0.04
                    elif vertical_speed_rec >= -3.0 and vertical_speed_rec <= -2.0:
                        reward += 0.01
                    else:
                        reward -= 0.08
            elif abs(alt_diff_ft) >= 500 and abs(alt_diff_ft) < 1000:
                if vertical_speed_rec >= 0: #if Above and climbing: WRONG
                    reward -= abs(vertical_speed_rec) / 20          #penalize wrong vertical trend 
                    if vertical_speed_rec < 3.0:
                        reward += 0.02
                    else:
                        reward -= 0.05                              #punish aggresive neg elev, or any pos eleva
                else:   #if above ane descending: RIGHT
                    if vertical_speed_rec > -4.0:
                        reward += 0.04
                    elif vertical_speed_rec >= -5.0 and vertical_speed_rec <= -4.0:
                        reward += 0.01
                    else:
                        reward -= 0.08
            else: #abs(alt_diff_ft) >= 1000:
                if vertical_speed_rec >= 0: #if above and climbing:WRONG
                    reward -= abs(vertical_speed_rec) / 15          #penalize wrong vertical trend 
                    if vertical_speed_rec < 4.0:    
                        reward += 0.02
                    else:
                        reward -= 0.05                              
                else: #if above and descending: RIGHT
                    if vertical_speed_rec > -5.0:
                        reward += 0.04
                    elif vertical_speed_rec >= -6.0 and vertical_speed_rec <= -5.0: 
                        reward += 0.01
                    else:
                        reward -= 0.08
        '''
        info = {}
        return obs, float(reward), terminated, truncated, info
    #ok given my current eval best csv, evaluate the plane attitude, what needs improvement
    '''
        #Air Speed policy 
        curr_ias_knots = self.fdm['velocities/vc-fps'] * 0.592484
        if curr_ias_knots < 140.0:
            reward -= 0.5
        elif curr_ias_knots >= 140.0 and curr_ias_knots < 180.0:
            reward -= 0.3
        elif curr_ias_knots >= 180.0 and curr_ias_knots < 250.0:
            reward -= 0.15
        elif curr_ias_knots >= 350.0:
            reward -= 0.15
        else: # between 250 and 350
            reward += 0.04

        #Elevator saturation / smoothness limiter
        delta_elev = abs(action[1] - self.prev_elev)
        reward -= 0.02 * delta_elev
        self.prev_elev = action[1]
        
        #vertical speed policy: reward agent if its adjusting smoothly towards the RIGHT direction 
        terminate_min_alt = 100
        terminate_max_alt = 30000
        alt_diff_vt = (curr_alt_ft - target_alt_ft)
        if alt_diff_vt < 0 and alt_diff_vt > -1000:         #below target altitude 
            if vertical_speed_rec > 0 and vertical_speed_rec < 2.0:     # reward positive smooth climb, punish 
                reward += 0.05
            else:
                reward -= 0.05      
        elif alt_diff_vt <= -1000 and alt_diff_vt > -5000:        
            if vertical_speed_rec > 0 and vertical_speed_rec < 4.0:     
                reward += 0.05
            else:
                reward -= 0.10      
        elif alt_diff_vt <= -5000:        
            if vertical_speed_rec > 0 and vertical_speed_rec < 6.0:     
                reward += 0.05
            else:
                reward -= 0.15  

        if alt_diff_vt > 0 and alt_diff_vt < 1000:         #below target altitude 
            if vertical_speed_rec < 0 and vertical_speed_rec > -2.0:     # reward positive smooth climb, punish 
                reward += 0.05
            else:
                reward -= 0.05      
        elif alt_diff_vt >= 1000 and alt_diff_vt < 5000:        
            if vertical_speed_rec < 0 and vertical_speed_rec > -4.0:     
                reward += 0.05
            else:
                reward -= 0.10      
        elif alt_diff_vt >= 5000:        
            if vertical_speed_rec < 0 and vertical_speed_rec > -6.0:     
                reward += 0.05
            else:
                reward -= 0.15  

        #rewards computations
        
        reward guides
        base survival reward:
        +0.05 to +0.1 per step

        crash:
        -100 is fine

        altitude error:
        1000 ft off -> about -0.1 to -0.2
        5000 ft off -> about -0.5 to -1.0

        near target altitude:
        within 50-100 ft -> small bonus, about +0.05 to +0.1

        correct vertical direction:
        tiny bonus only, about +0.02 to +0.05

        wrong vertical direction:
        penalty, about -0.05 to -0.15

        too much vertical speed:
        penalty, about -0.1 to -0.5 depending how extreme

        smooth near target:
        bonus, about +0.05 to +0.1
    '''
        
    

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
