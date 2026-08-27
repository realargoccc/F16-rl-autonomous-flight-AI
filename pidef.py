#defensive environment — everything inherited from F16Env except _reward
#obs / physics / actions / termination must stay identical or the selector can't hand over mid-episode

import math
import numpy as np
from flight_env import F16Env, RewardOut

FOE_POOL = ["v2.8.0", "v2.8.1", "v2.8.2", "v2.8.3", "v2.8.4",
            "v2.8.5", "v2.8.6", "v2.8.7", "v2.8.8", "v2.8.9"]


class F16DefEnv(F16Env):
    def __init__(self):
        super().__init__()

        #reset() only flips side when foe_policy is not None, so the pool prob must be 1.0
        self.defensive_p = 1.0
        self.foe_pool_prob = 1.0

        #reward coefficients
        self.k_bridge = 300.0
        self.r_survive = 0.0

    def _reward(self, action, speed_knots, curr_g, alt_agl_m, dt, crashed, foe_crashed, deck_hit, truncated, dmg_foe, dmg_me):
        reward = 0.0
        pot = 0.0

        #constraint rails: stall speed and g loads
        r_rails = 0.0
        if speed_knots < 150:
            r_rails -= 0.01 * (150 - speed_knots)
        if curr_g > 9.0:
            r_rails -= 0.5 * (curr_g - 9.0) ** 2
        elif curr_g < -1.0:
            r_rails -= 0.5 * (-1.0 - curr_g) ** 2

        #hard deck
        r_deck = 0.0
        alt_agl_kft = alt_agl_m / 304.8
        if alt_agl_kft < 6.0:
            r_deck -= 0.05 * (6.0 - alt_agl_kft) ** 2

        #wez 
        r_wez = self.k_damage * (dmg_foe - dmg_me)

        #positional advantage
        foe_boresight = self.foe.boresight_to(self.me.pos())
        eta_ata = 1.0 - self.boresight / np.pi
        eta_aa  = self.aspect_angle / np.pi
        agent_adv = 0.5 * eta_ata + 0.5 * eta_aa

        foe_eta_ata = 1.0 - foe_boresight / np.pi
        foe_eta_aa = self.foe_state.aspect_angle / np.pi
        foe_adv = 0.5 * foe_eta_ata + 0.5 * foe_eta_aa

        r_adv = 0.5 * (agent_adv - foe_adv)
        pot = self.k_bridge * min(agent_adv - foe_adv, 0.0)

        #terminal conditions
        r_terminal = 0.0
        if truncated: r_terminal += self.r_survive
        if deck_hit:  r_terminal -= 300.0
        if crashed:   r_terminal -= 300.0
        if self.foe_hp   - dmg_foe <= 0.0: r_terminal += 400.0
        if self.agent_hp - dmg_me  <= 0.0: r_terminal -= 400.0

        self.last_terms = {"rails": r_rails, "deck": r_deck, "wez": r_wez,
                           "adv": r_adv, "term": r_terminal, "pot": pot}
        reward = r_rails + r_deck + r_wez + r_adv + r_terminal
        return RewardOut(reward, dmg_foe, dmg_me, pot)


def make_def_env(pool=FOE_POOL):
    env = F16DefEnv()
    for tag in pool:
        env.load_foe(tag)
    return env


if __name__ == "__main__":
    #parity: same seed, same actions, obs and termination must match flight_env exactly
    a, b = F16Env(), F16DefEnv()
    b.defensive_p = 0.0
    b.foe_pool_prob = 0.0

    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    rng = np.random.default_rng(0)
    for k in range(300):
        bad = np.flatnonzero(oa != ob)
        if bad.size:
            raise SystemExit("step %d: obs differ at %s" % (k, bad.tolist()))
        act = rng.uniform(-1, 1, 4).astype(np.float32)
        oa, _, ta, ua, _ = a.step(act)
        ob, _, tb, ub, _ = b.step(act)
        if (ta, ua) != (tb, ub):
            raise SystemExit("step %d: termination differs" % k)
        if ta or ua:
            break
    print("parity OK — %d steps" % (k + 1))

    e = make_def_env()
    e.reset(seed=7)
    print("setup %s   defensive_p %.1f   pool %d" % (e.setup, e.defensive_p, len(e.foe_pool)))