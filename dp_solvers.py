# -*- coding: utf-8 -*-
"""
Functions to solve the MDP using dynamic programming.
"""
import numpy as np
from robot_mdp import Params, H, L, SEARCH, WAIT, RECHARGE

def value_iteration(p: Params, tol=1e-12, max_iter=100000):
    """
    Computes the optimal values v*(H), v*(L) and the greedy optimal policy
    using value iteration.
    """
    vH = 0.0
    vL = 0.0
    for _ in range(max_iter):
        qH_search = p.r_search + p.gamma * (p.alpha_p * vH + (1 - p.alpha_p) * vL)
        qH_wait = p.r_wait + p.gamma * vH
        vH_new = max(qH_search, qH_wait)

        qL_search = (p.r_search * p.beta_p + p.penalty * (1 - p.beta_p)
                     + p.gamma * (p.beta_p * vL + (1 - p.beta_p) * vH))
        qL_wait = p.r_wait + p.gamma * vL
        qL_recharge = 0.0 + p.gamma * vH
        vL_new = max(qL_search, qL_wait, qL_recharge)

        if abs(vH_new - vH) + abs(vL_new - vL) < tol:
            vH, vL = vH_new, vL_new
            break
        vH, vL = vH_new, vL_new

    # Derive greedy optimal policy from the converged values
    qH_search = p.r_search + p.gamma * (p.alpha_p * vH + (1 - p.alpha_p) * vL)
    qH_wait = p.r_wait + p.gamma * vH
    aH = SEARCH if qH_search >= qH_wait else WAIT

    qL_search = (p.r_search * p.beta_p + p.penalty * (1 - p.beta_p)
                 + p.gamma * (p.beta_p * vL + (1 - p.beta_p) * vH))
    qL_wait = p.r_wait + p.gamma * vL
    qL_recharge = 0.0 + p.gamma * vH
    aL = [SEARCH, WAIT, RECHARGE][int(np.argmax([qL_search, qL_wait, qL_recharge]))]

    return (vH, vL), {H: aH, L: aL}