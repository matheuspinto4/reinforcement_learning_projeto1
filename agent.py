# -*- coding: utf-8 -*-
"""
Defines the reinforcement learning agent class
that uses TD Control (SARSA or Q-learning).
"""
import numpy as np
from robot_mdp import Params, H, L, SEARCH, WAIT, RECHARGE, valid_actions

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
        self.Q = np.zeros((2, 3), dtype=float)
        self.rng = np.random.default_rng(seed)

    def act(self, state: int):
        """ε-greedy action selection restricted to valid actions."""
        acts = valid_actions(state)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(acts)
        qs = np.array([self.Q[state, a] for a in acts])
        a_idx = int(np.argmax(qs + 1e-9 * self.rng.standard_normal(qs.shape)))
        return acts[a_idx]

    def improve(self, s, a, r, s2):
        """A single TD update (SARSA or Q-learning)."""
        if self.method == "sarsa":
            a2 = self.act(s2)
            target = r + self.params.gamma * self.Q[s2, a2]
        else:
            acts2 = valid_actions(s2)
            target = r + self.params.gamma * np.max(self.Q[s2, acts2])

        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay(self):
        """Decays the epsilon value for exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_policy(self, path="policy_robot_q.npy"):
        np.save(path, self.Q)

    def load_policy(self, path="policy_robot_q.npy"):
        self.Q = np.load(path)

    def policy_probs(self, epsilon_eval=0.0):
        """
        Calculates the policy probability matrix π(a|s)
        based on the Q-table.
        """
        P = np.zeros((2, 3), dtype=float)
        for s in (H, L):
            acts = valid_actions(s)
            qs = np.array([self.Q[s, a] for a in acts])
            maxq = np.max(qs)
            greedy_mask = (qs >= maxq - 1e-12)
            n_greedy = int(np.sum(greedy_mask))
            k = len(acts)
            eps = float(epsilon_eval)
            base = eps / k
            P_row = np.full(k, base, dtype=float)
            P_row[greedy_mask] += (1.0 - eps) / n_greedy
            for idx, a in enumerate(acts):
                P[s, a] = P_row[idx]
        return P