# -*- coding: utf-8 -*-
"""
Functions for analyzing and visualizing the training results and policy.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from robot_mdp import Params, H, L, SEARCH, WAIT, RECHARGE, STATE_NAMES, ACTION_NAMES
from agent import Agent

# Optional seaborn for heatmap
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False


def greedy_policy_from_Q(Q):
    """Extracts the greedy policy from a Q-table."""
    pi = {}
    pi[H] = [SEARCH, WAIT][int(np.argmax(Q[H, [SEARCH, WAIT]]))]
    pi[L] = [SEARCH, WAIT, RECHARGE][int(np.argmax(Q[L, [SEARCH, WAIT, RECHARGE]]))]
    return pi


def transition_reward_under_policy(p: Params, policy):
    """Calculates the transition and reward tables for a fixed policy."""
    P = np.zeros((2,2)); R = np.zeros(2)
    # H
    if policy[H] == SEARCH:
        P[H,H] = p.alpha_p; P[H,L] = 1 - p.alpha_p; R[H] = p.r_search
    else:  # WAIT
        P[H,H] = 1.0; R[H] = p.r_wait
    # L
    a = policy[L]
    if a == SEARCH:
        P[L,L] = p.beta_p; P[L,H] = 1 - p.beta_p
        R[L] = p.r_search * p.beta_p + p.penalty * (1 - p.beta_p)
    elif a == WAIT:
        P[L,L] = 1.0; R[L] = p.r_wait
    else:  # RECHARGE
        P[L,H] = 1.0; R[L] = 0.0
    return P, R


def stationary_distribution(P):
    """Calculates the stationary distribution of a transition matrix."""
    A = (P.T - np.eye(2))
    A = np.vstack([A, np.ones(2)])
    b = np.array([0.0, 0.0, 1.0])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def avg_reward_of_policy(p: Params, policy):
    """Calculates the average reward per step for a fixed policy."""
    P, R = transition_reward_under_policy(p, policy)
    pi = stationary_distribution(P)
    return float(pi @ R)


def plot_epoch_metrics(rewards_path="rewards.txt",
                       steps_per_epoch=1000,
                       Q_path="policy_robot_q_run1.npy",
                       params=None,
                       window=10,
                       title_suffix=""):
    """
    Plots the total reward per epoch and the moving average.
    """
    if not os.path.exists(rewards_path):
        print(f"File not found: {rewards_path}")
        return
    r = np.loadtxt(rewards_path, dtype=float)

    if window and window > 1:
        kernel = np.ones(window) / window
        mov = np.convolve(r, kernel, mode="valid")
    else:
        mov = None

    plt.figure()
    plt.plot(r, label="per-epoch total")
    if mov is not None:
        x = np.arange(window-1, window-1 + len(mov))
        plt.plot(x, mov, label=f"moving avg (w={window})", linestyle="--")

    if params is not None and os.path.exists(Q_path):
        Q = np.load(Q_path)
        pi = greedy_policy_from_Q(Q)
        g = avg_reward_of_policy(params, pi)
        target = g * steps_per_epoch
        plt.axhline(target, linestyle=":", label=f"target ≈ {target:.2f} (g={g:.3f})")

    plt.xlabel("Epoch")
    plt.ylabel("Total reward (epoch)")
    plt.title(f"Per-epoch reward {title_suffix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    per_step = r / float(steps_per_epoch)
    if mov is not None:
        mov_per_step = mov / float(steps_per_epoch)
    plt.figure()
    plt.plot(per_step, label="per-step average (epoch total / steps)")
    if mov is not None:
        plt.plot(x, mov_per_step, label=f"moving avg per-step (w={window})", linestyle="--")
    if params is not None and os.path.exists(Q_path):
        plt.axhline(g, linestyle=":", label=f"target g ≈ {g:.3f}")
    plt.xlabel("Epoch")
    plt.ylabel("Average reward per step")
    plt.title(f"Per-step average reward {title_suffix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_policy_heatmap(Q_path="policy_robot_q_run1.npy", epsilon_eval=0.05):
    """
    Shows a heatmap of the policy π(a|s) extracted from the Q-table.
    """
    if not os.path.exists(Q_path):
        print(f"Policy file not found: {Q_path}")
        return
    Q = np.load(Q_path)
    dummy_params = Params()
    agent = Agent(dummy_params, method="qlearning")
    agent.Q = Q
    P = agent.policy_probs(epsilon_eval=epsilon_eval)

    y_labels = [f"State {STATE_NAMES[s]}" for s in (H, L)]
    x_labels = [ACTION_NAMES[a] for a in (SEARCH, WAIT, RECHARGE)]

    if HAS_SNS:
        mask = np.zeros_like(P, dtype=bool)
        mask[H, RECHARGE] = True
        plt.figure()
        sns.heatmap(P, annot=True, fmt=".2f", xticklabels=x_labels,
                    yticklabels=y_labels, mask=mask, cbar=True)
        plt.title(f"Policy heatmap π(a|s) (ε_eval={epsilon_eval})")
        plt.xlabel("Action")
        plt.ylabel("State")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        plt.imshow(P, aspect="auto")
        plt.colorbar()
        plt.xticks(range(3), x_labels)
        plt.yticks(range(2), y_labels)
        plt.title(f"Policy heatmap π(a|s) (ε_eval={epsilon_eval})")
        plt.tight_layout()
        plt.show()