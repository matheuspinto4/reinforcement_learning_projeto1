# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 17:50:39 2025

@author: laio_
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
import time

# Optional seaborn for heatmap
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# ------------------ MDP (Sutton Example 3.3) ------------------

# States
H, L = 0, 1
STATE_NAMES = {H: "H", L: "L"}

# Actions
SEARCH, WAIT, RECHARGE = 0, 1, 2
ACTION_NAMES = {SEARCH: "search", WAIT: "wait", RECHARGE: "recharge"}

# Valid actions by state
ACTIONS_H = (SEARCH, WAIT)                # in H you cannot recharge
ACTIONS_L = (SEARCH, WAIT, RECHARGE)


# ---- constants at top ----
STEPS_PER_EPOCH = 2000
EPOCHS = 500
REPEAT_RUNS = 5

@dataclass
class Params:
    alpha_p: float = 0.9
    beta_p: float  = 0.9
    r_search: float = 1.0
    r_wait: float   = 0.0
    penalty: float  = -3.0
    gamma: float    = 0.9
    def __post_init__(self):
        assert self.r_search > self.r_wait, "Spec requires r_search > r_wait"
        assert 0.0 <= self.gamma < 1.0, "Discount factor gamma must be in [0,1)"
        assert 0.0 <= self.alpha_p <= 1.0 and 0.0 <= self.beta_p <= 1.0


def env_step(state: int, action: int, p: Params, rng: np.random.Generator):
    """Sample (next_state, reward) from Sutton's recycling robot MDP."""
    if state == H:
        if action == SEARCH:
            # Always get r_search; remain H with prob alpha_p, else go to L
            rew = p.r_search
            next_state = H if rng.random() < p.alpha_p else L
            return next_state, rew
        elif action == WAIT:
            # Stay in H, reward r_wait
            return H, p.r_wait
        else:
            raise ValueError("RECHARGE is invalid in state H")
    else:  # state == L
        if action == SEARCH:
            # With prob beta_p: stay L, get r_search
            # With prob 1 - beta_p: deplete -> go to H with penalty
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
    return ACTIONS_H if state == H else ACTIONS_L

# ------------------ Agent: TD Control (SARSA or Q-learning) ------------------

class Agent:
    def __init__(self, params: Params,
                 alpha=0.2, epsilon=0.2, epsilon_min=0.01, epsilon_decay=0.9995,
                 method="qlearning", seed=0):
        """
        method: 'sarsa' (on-policy) or 'qlearning' (off-policy)
        """
        self.params = params
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.method = method.lower()
        assert self.method in ("sarsa", "qlearning")
        # Q-table for 2 states x 3 actions; invalid actions kept but never chosen
        self.Q = np.zeros((2, 3), dtype=float)
        self.rng = np.random.default_rng(seed)

    def act(self, state: int):
        """ε-greedy action selection restricted to valid actions."""
        acts = valid_actions(state)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(acts)
        # exploit: argmax Q(s,a) over valid actions; break ties randomly
        qs = np.array([self.Q[state, a] for a in acts])
        # random tie-break: add tiny noise
        a_idx = int(np.argmax(qs + 1e-9 * self.rng.standard_normal(qs.shape)))
        return acts[a_idx]

    def improve(self, s, a, r, s2):
        """One TD update (SARSA or Q-learning)."""
        if self.method == "sarsa":
            # on-policy: next action follows current ε-greedy policy
            a2 = self.act(s2)
            target = r + self.params.gamma * self.Q[s2, a2]
        else:
            # Q-learning: bootstrap with max_a' Q(s', a')
            acts2 = valid_actions(s2)
            target = r + self.params.gamma * np.max(self.Q[s2, acts2])

        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --- utilities to persist / inspect policy ---
    def save_policy(self, path="policy_robot_q.npy"):
        np.save(path, self.Q)

    def load_policy(self, path="policy_robot_q.npy"):
        self.Q = np.load(path)

    def policy_probs(self, epsilon_eval=0.0):
        """
        Build π(a|s) matrix (2x3) under ε-greedy w.r.t. Q with evaluation epsilon.
        Invalid actions get probability 0.
        """
        P = np.zeros((2, 3), dtype=float)
        for s in (H, L):
            acts = valid_actions(s)
            qs = np.array([self.Q[s, a] for a in acts])
            # Find greedy set (may be ties)
            maxq = np.max(qs)
            greedy_mask = (qs >= maxq - 1e-12)
            n_greedy = int(np.sum(greedy_mask))
            k = len(acts)
            eps = float(epsilon_eval)

            # ε-greedy distribution: split epsilon uniformly, and put (1-ε) on greedy set (split if tie)
            base = eps / k
            P_row = np.full(k, base, dtype=float)
            P_row[greedy_mask] += (1.0 - eps) / n_greedy

            # write back into the 2x3 matrix at action columns
            for idx, a in enumerate(acts):
                P[s, a] = P_row[idx]
        return P

# ------------------ Training, Logging, and Plots ------------------

def train_td(epochs=50, steps_per_epoch=1000,
             params=Params(alpha_p=0.9, beta_p=0.9, r_search=1.0, r_wait=0.0, penalty=-3.0, gamma=0.9),
             alpha=0.2, epsilon=0.3, epsilon_min=0.01, epsilon_decay=0.9995,
             method="qlearning", seed=42, rewards_file="rewards.txt",
             repeat_runs=1, avg_out="rewards_mean.txt"):
    """
    Trains for 'epochs', each epoch = 'steps_per_epoch' transitions.
    Saves total reward per epoch to rewards.txt (one number per line).
    If repeat_runs > 1, averages runs and saves to rewards_mean.txt.
    """
    all_runs = []
    for run in range(repeat_runs):
        agent = Agent(params, alpha, epsilon, epsilon_min, epsilon_decay, method, seed + run)
        rng = np.random.default_rng(seed + 10_000 + run)
        rewards_per_epoch = []

        for ep in range(1, epochs + 1):
            s = H  # start from High battery every epoch (common choice)
            G = 0.0
            # Optional: sample a random start: s = int(rng.random() < 0.5)
            for _ in range(steps_per_epoch):
                a = agent.act(s)
                s2, r = env_step(s, a, params, rng)
                agent.improve(s, a, r, s2)
                s = s2
                G += r
            agent.decay()
            rewards_per_epoch.append(G)
            if ep % max(1, (epochs // 10)) == 0:
                print(f"[run {run+1}/{repeat_runs}] epoch {ep}/{epochs} | total reward={G:.3f} | eps={agent.epsilon:.3f}", flush=True)

        all_runs.append(np.array(rewards_per_epoch, dtype=float))
        # Save last run's Q for inspection
        agent.save_policy(f"policy_robot_q_run{run+1}.npy")

    # Save rewards.txt (by spec) for the LAST run
    np.savetxt(rewards_file, all_runs[-1], fmt="%.6f")

    # If multiple runs, also save mean across runs
    if repeat_runs > 1:
        mean_curve = np.mean(np.vstack(all_runs), axis=0)
        np.savetxt(avg_out, mean_curve, fmt="%.6f")
        print(f"Saved mean curve to {avg_out}", flush=True)

    print(f"Saved last-run rewards to {rewards_file}", flush=True)
    return all_runs[-1] if repeat_runs == 1 else np.mean(np.vstack(all_runs), axis=0)


# --- helpers to compute a target line from the learned greedy policy ---
#H, L = 0, 1
#SEARCH, WAIT, RECHARGE = 0, 1, 2

def greedy_policy_from_Q(Q):
    pi = {}
    pi[H] = [SEARCH, WAIT][int(np.argmax(Q[H, [SEARCH, WAIT]]))]
    pi[L] = [SEARCH, WAIT, RECHARGE][int(np.argmax(Q[L, [SEARCH, WAIT, RECHARGE]]))]
    return pi

def transition_reward_under_policy(p, policy):
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
    A = (P.T - np.eye(2))
    A = np.vstack([A, np.ones(2)])
    b = np.array([0.0, 0.0, 1.0])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi

def avg_reward_of_policy(p, policy):
    P, R = transition_reward_under_policy(p, policy)
    pi = stationary_distribution(P)
    return float(pi @ R)

# --- main plot (no cumulative) ---
def plot_epoch_metrics(rewards_path="rewards.txt",
                       steps_per_epoch=1000,
                       Q_path="policy_robot_q_run1.npy",
                       params=None,
                       window=10,
                       title_suffix=""):
    """
    Plots:
      1) per-epoch total reward
      2) moving average (window)
      3) per-step average reward (total/steps)
      4) horizontal target line ≈ g * steps_per_epoch (if Q & params provided)
    """
    if not os.path.exists(rewards_path):
        print(f"File not found: {rewards_path}")
        return
    r = np.loadtxt(rewards_path, dtype=float)

    # moving average
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

    # target line from learned greedy policy (optional)
    if params is not None and os.path.exists(Q_path):
        Q = np.load(Q_path)
        pi = greedy_policy_from_Q(Q)
        g = avg_reward_of_policy(params, pi)          # avg reward per step
        target = g * steps_per_epoch
        plt.axhline(target, linestyle=":", label=f"target ≈ {target:.2f} (g={g:.3f})")

    plt.xlabel("Epoch")
    plt.ylabel("Total reward (epoch)")
    plt.title(f"Per-epoch reward {title_suffix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # also show per-step average (bounded, should stabilize near g)
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
    Shows a 2x3 heatmap of π(a|s) under ε-greedy w.r.t. loaded Q.
    Each cell = probability of taking action a in state s at evaluation time.
    Invalid actions appear as 0 (or are masked if seaborn is available).
    """
    if not os.path.exists(Q_path):
        print(f"Policy file not found: {Q_path}")
        return
    Q = np.load(Q_path)
    dummy_params = Params()
    agent = Agent(dummy_params, method="qlearning")
    agent.Q = Q
    P = agent.policy_probs(epsilon_eval=epsilon_eval)  # 2x3 matrix

    # Prepare labels
    y_labels = [f"State {STATE_NAMES[s]}" for s in (H, L)]
    x_labels = [ACTION_NAMES[a] for a in (SEARCH, WAIT, RECHARGE)]

    if HAS_SNS:
        mask = np.zeros_like(P, dtype=bool)
        # Mask invalid action 'recharge' in H
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
        # Fallback to plain matplotlib
        plt.figure()
        plt.imshow(P, aspect="auto")
        plt.colorbar()
        plt.xticks(range(3), x_labels)
        plt.yticks(range(2), y_labels)
        plt.title(f"Policy heatmap π(a|s) (ε_eval={epsilon_eval})")
        plt.tight_layout()
        plt.show()

# ---- Transition & reward tables for visualization ----
def build_P_R_tables(p):
    P_sa = np.full((2, 3, 2), np.nan, dtype=float)
    R_sa = np.full((2, 3), np.nan, dtype=float)

    # H, search
    P_sa[H, SEARCH, H] = p.alpha_p
    P_sa[H, SEARCH, L] = 1 - p.alpha_p
    R_sa[H, SEARCH]    = p.r_search

    # H, wait
    P_sa[H, WAIT, H] = 1.0
    P_sa[H, WAIT, L] = 0.0
    R_sa[H, WAIT]    = p.r_wait

    # L, search
    P_sa[L, SEARCH, L] = p.beta_p
    P_sa[L, SEARCH, H] = 1 - p.beta_p
    R_sa[L, SEARCH]    = p.r_search * p.beta_p + p.penalty * (1 - p.beta_p)

    # L, wait
    P_sa[L, WAIT, L] = 1.0
    P_sa[L, WAIT, H] = 0.0
    R_sa[L, WAIT]    = p.r_wait

    # L, recharge
    P_sa[L, RECHARGE, H] = 1.0
    P_sa[L, RECHARGE, L] = 0.0
    R_sa[L, RECHARGE]    = 0.0

    return P_sa, R_sa

def plot_sa_to_sprime_heatmap(P_sa):
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

def steps_to_threshold(rewards_per_epoch, steps_per_epoch, g_target, eps=0.01, window=10):
    per_step = rewards_per_epoch / float(steps_per_epoch)
    if len(per_step) < window: return None
    mov = np.convolve(per_step, np.ones(window)/window, mode="valid")
    idxs = np.where(mov >= (g_target - eps))[0]
    if len(idxs) == 0: return None
    reached_epoch = idxs[0] + window      # first epoch index that meets target (window end)
    return reached_epoch * steps_per_epoch


# States/actions must already be defined:

def value_iteration(p, tol=1e-12, max_iter=100000):
    """
    Compute optimal values v*(H), v*(L) and a greedy optimal policy for the
    recycling robot MDP (Sutton Ex. 3.3) given Params p.
    Returns: ((vH, vL), {H: aH, L: aL})
    """
    vH = 0.0
    vL = 0.0
    for _ in range(max_iter):
        # Qs at H
        qH_search = p.r_search + p.gamma * (p.alpha_p * vH + (1 - p.alpha_p) * vL)
        qH_wait   = p.r_wait   + p.gamma * vH
        vH_new = max(qH_search, qH_wait)

        # Qs at L
        qL_search   = (p.r_search * p.beta_p + p.penalty * (1 - p.beta_p)
                       + p.gamma * (p.beta_p * vL + (1 - p.beta_p) * vH))
        qL_wait     = p.r_wait + p.gamma * vL
        qL_recharge = 0.0      + p.gamma * vH
        vL_new = max(qL_search, qL_wait, qL_recharge)

        if abs(vH_new - vH) + abs(vL_new - vL) < tol:
            vH, vL = vH_new, vL_new
            break
        vH, vL = vH_new, vL_new

    # Derive greedy optimal policy from the converged values
    qH_search = p.r_search + p.gamma * (p.alpha_p * vH + (1 - p.alpha_p) * vL)
    qH_wait   = p.r_wait   + p.gamma * vH
    aH = SEARCH if qH_search >= qH_wait else WAIT

    qL_search   = (p.r_search * p.beta_p + p.penalty * (1 - p.beta_p)
                   + p.gamma * (p.beta_p * vL + (1 - p.beta_p) * vH))
    qL_wait     = p.r_wait + p.gamma * vL
    qL_recharge = 0.0      + p.gamma * vH
    aL = [SEARCH, WAIT, RECHARGE][int(np.argmax([qL_search, qL_wait, qL_recharge]))]

    return (vH, vL), {H: aH, L: aL}


def train_with_metrics(params, epochs, steps_per_epoch, agent_kwargs, seed=42,
                       rewards_file="rewards.txt", q_out="policy_robot_q_run1.npy",
                       dp_value_iteration_fn=None):
    # dp_value_iteration_fn should return ((vH,vL), pi_opt)
    t0 = time.perf_counter()
    _ = train_td(epochs=epochs, steps_per_epoch=steps_per_epoch,
                 params=params, seed=seed, rewards_file=rewards_file, **agent_kwargs)
    t1 = time.perf_counter()
    time_sec = t1 - t0

    # load rewards + learned Q
    r = np.loadtxt(rewards_file, dtype=float)
    Q = np.load(q_out)
    pi_learned = greedy_policy_from_Q(Q)
    g_learned = avg_reward_of_policy(params, pi_learned)

    # DP-optimal target (optional but recommended)
    g_opt = None; steps_eps = None
    if dp_value_iteration_fn is not None:
        (_, _), pi_opt = dp_value_iteration_fn(params)
        g_opt = avg_reward_of_policy(params, pi_opt)
        steps_eps = steps_to_threshold(r, steps_per_epoch, g_opt, eps=0.01, window=10)

    # simple efficiencies
    eff_return_per_sec = g_learned / max(time_sec, 1e-9)
    eff_return_per_step = g_learned  # since g is already per step

    return {
        "time_sec": time_sec,
        "g_learned": g_learned,
        "g_opt": g_opt,
        "gap": (None if g_opt is None else g_opt - g_learned),
        "steps_to_eps_opt": steps_eps,
        "eff_return_per_sec": eff_return_per_sec,
        "eff_return_per_step": eff_return_per_step,
    }


if __name__ == "__main__":
    params = Params(alpha_p=0.5, beta_p=0.5, r_search=1.0, r_wait=0.5, penalty=-3.0, gamma=0.5)

    
    agent_kwargs = dict(alpha=0.2, epsilon=0.3, epsilon_min=0.01, epsilon_decay=0.99, method="qlearning")
    metrics = train_with_metrics(
        params,
        epochs=60,
        steps_per_epoch=STEPS_PER_EPOCH,
        agent_kwargs=agent_kwargs,
        seed=42,
        dp_value_iteration_fn=value_iteration
        )
    print(metrics)

    _ = train_td(
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        params=params,
        alpha=0.2,
        epsilon=0.3,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        method="qlearning",
        seed=42,
        rewards_file="rewards.txt",
        repeat_runs=REPEAT_RUNS,
        avg_out="rewards_mean.txt"
    )

    plot_epoch_metrics(
        rewards_path="rewards.txt",
        steps_per_epoch=STEPS_PER_EPOCH,
        Q_path=f"policy_robot_q_run{REPEAT_RUNS}.npy",
        params=params,
        window=10,
        title_suffix=f"(αp={params.alpha_p}, βp={params.beta_p}, γ={params.gamma})"
    )

    plot_policy_heatmap(Q_path=f"policy_robot_q_run{REPEAT_RUNS}.npy", epsilon_eval=0.05)

    P_sa, R_sa = build_P_R_tables(params)
    plot_sa_to_sprime_heatmap(P_sa)
    plot_expected_reward_heatmap(R_sa)

    

