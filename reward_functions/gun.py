from .base import BaseReward

class Gun(BaseReward):
    '''damage pricing only rewards, mechanics in step of flight env'''

    def raw(self, env, computed):
        return env.k_damage * (computed.dmg_foe - computed.dmg_me)

    