from .base import BaseReward

class Terminal(BaseReward):
    '''win lose conditions'''
    def raw(self, env, computed):
        r = 0.0
        if computed.deck_hit: r -= env.k_crash
        if computed.crashed:  r -= env.k_crash
        if env.foe_hp - computed.dmg_foe <= 0.0: r += env.k_win
        if env.agent_hp - computed.dmg_me <= 0.0: r -= env.k_win
        return r