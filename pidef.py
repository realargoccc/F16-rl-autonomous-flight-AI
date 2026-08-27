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

    def _reward(self, action, speed_knots, curr_g, alt_agl_m, dt, crashed, deck_hit):
        reward = 0.0
        dmg_foe = 0.0
        dmg_me = 0.0
        pot = 0.0

        foe_boresight = self.foe.boresight_to(self.me.pos())

        # constraint rails

        #below deck punishment

        #punish huge oscillation

        #wez agent's configs — must set dmg_foe, step() subtracts it

        #wez bandit's configs — must set dmg_me, step() subtracts it

        #positional advantage - ATA and AA

        #distance away — measured -125/ep on defensive spawns, decide if this quadrant carries it

        #terminals — crashed / deck_hit come in already evaluated, don't recompute
        #win/lose read from (hp - dmg) here, step() re-reads them after subtracting

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