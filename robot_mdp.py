# -*- coding: utf-8 -*-
"""
Defines the Recycling Robot MDP environment.
Based on Example 3.3 from the Sutton & Barto textbook.
"""
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import os

# Optional seaborn for heatmap
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# States
H, L = 0, 1
STATE_NAMES = {H: "H", L: "L"}

# Actions
SEARCH, WAIT, RECHARGE = 0, 1, 2
ACTION_NAMES = {SEARCH: "search", WAIT: "wait", RECHARGE: "recharge"}

# Valid actions by state
ACTIONS_H = (SEARCH, WAIT)
ACTIONS_L = (SEARCH, WAIT, RECHARGE)

@dataclass
class Params:
    """Parameters for the MDP environment."""
    alpha_p: float = 0.9
    beta_p: float = 0.9
    r_search: float = 1.0
    r_wait: float = 0.0
    penalty: float = -3.0
    gamma: float = 0.9
    def __post_init__(self):
        assert self.r_search > self.r_wait, "Spec requires r_search > r_wait"
        assert 0.0 <= self.gamma < 1.0, "Discount factor gamma must be in [0,1)"
        assert 0.0 <= self.alpha_p <= 1.0 and 0.0 <= self.beta_p <= 1.0


def env_step(state: int, action: int, p: Params, rng: np.random.Generator):
    """Sample (next_state, reward) from the robot MDP."""
    if state == H:
        if action == SEARCH:
            rew = p.r_search
            next_state = H if rng.random() < p.alpha_p else L
            return next_state, rew
        elif action == WAIT:
            return H, p.r_wait
        else:
            raise ValueError("RECHARGE is invalid in state H")
    else:  # state == L
        if action == SEARCH:
            if rng.random() < p.beta_p:
                return L, p.r_search
            else:
                return H, p.penalty
        elif action == WAIT:
            return L, p.r_wait
        elif action == RECHARGE:
            return H, 0.0
        else:
            raise ValueError("Invalid action code")


def valid_actions(state: int):
    """Returns the tuple of valid actions for a given state."""
    return ACTIONS_H if state == H else ACTIONS_L


def build_P_R_tables(p: Params):
    """Builds transition P(s'|s,a) and reward R(s,a) tables."""
    P_sa = np.full((2, 3, 2), np.nan, dtype=float)
    R_sa = np.full((2, 3), np.nan, dtype=float)

    # H, search
    P_sa[H, SEARCH, H] = p.alpha_p
    P_sa[H, SEARCH, L] = 1 - p.alpha_p
    R_sa[H, SEARCH] = p.r_search

    # H, wait
    P_sa[H, WAIT, H] = 1.0
    P_sa[H, WAIT, L] = 0.0
    R_sa[H, WAIT] = p.r_wait

    # L, search
    P_sa[L, SEARCH, L] = p.beta_p
    P_sa[L, SEARCH, H] = 1 - p.beta_p
    R_sa[L, SEARCH] = p.r_search * p.beta_p + p.penalty * (1 - p.beta_p)

    # L, wait
    P_sa[L, WAIT, L] = 1.0
    P_sa[L, WAIT, H] = 0.0
    R_sa[L, WAIT] = p.r_wait

    # L, recharge
    P_sa[L, RECHARGE, H] = 1.0
    P_sa[L, RECHARGE, L] = 0.0
    R_sa[L, RECHARGE] = 0.0

    return P_sa, R_sa


def plot_sa_to_sprime_heatmap(P_sa):
    """Plots a heatmap of the transition probabilities."""
    col_pairs = [(H,SEARCH), (H,WAIT), (L,SEARCH), (L,WAIT), (L,RECHARGE)]
    mat = np.full((2, len(col_pairs)), np.nan, dtype=float)
    col_labels = []
    for j,(s,a) in enumerate(col_pairs):
        col_labels.append(f"{STATE_NAMES[s]}:{ACTION_NAMES[a]}")
        if np.isnan(P_sa[s, a]).all():
            continue
        mat[H, j] = P_sa[s, a, H]
        mat[L, j] = P_sa[s, a, L]

    plt.figure()
    if HAS_SNS:
        sns.heatmap(mat, annot=True, fmt=".2f",
                    xticklabels=col_labels, yticklabels=[f"s'={STATE_NAMES[H]}", f"s'={STATE_NAMES[L]}"],
                    cbar=True)
    else:
        plt.imshow(mat, aspect="auto"); plt.colorbar()
        plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
        plt.yticks([0,1], [f"s'={STATE_NAMES[H]}", f"s'={STATE_NAMES[L]}"])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i,j]
                if not np.isnan(v): plt.text(j, i, f"{v:.2f}", ha="center", va="center", color="w")
    plt.title("Transition probabilities P(s' | s,a)")
    plt.tight_layout(); plt.show()


def plot_expected_reward_heatmap(R_sa):
    """Plots a heatmap of the expected rewards."""
    col_pairs = [(H,SEARCH), (H,WAIT), (L,SEARCH), (L,WAIT), (L,RECHARGE)]
    mat = np.full((2, len(col_pairs)), np.nan, dtype=float)
    col_labels = []
    for j,(s,a) in enumerate(col_pairs):
        col_labels.append(f"{STATE_NAMES[s]}:{ACTION_NAMES[a]}")
        if np.isnan(R_sa[s, a]): continue
        mat[s, j] = R_sa[s, a]

    plt.figure()
    if HAS_SNS:
        sns.heatmap(mat, annot=True, fmt=".2f",
                    xticklabels=col_labels, yticklabels=[STATE_NAMES[H], STATE_NAMES[L]],
                    cmap="coolwarm", center=0.0, cbar=True)
    else:
        plt.imshow(mat, aspect="auto"); plt.colorbar()
        plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
        plt.yticks([0,1], [STATE_NAMES[H], STATE_NAMES[L]])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i,j]
                if not np.isnan(v): plt.text(j, i, f"{v:.2f}", ha="center", va="center", color="k")
    plt.title("Expected immediate reward E[r | s,a]")
    plt.tight_layout(); plt.show()