# -*- coding: utf-8 -*-
"""
Contains the training loop and functions to measure performance metrics.
"""
import numpy as np
import time
import os
from agent import Agent
from robot_mdp import Params, env_step, H, L
from analysis import avg_reward_of_policy, greedy_policy_from_Q

def train_td(epochs=50, steps_per_epoch=1000,
             params=Params(),
             alpha=0.2, epsilon=0.3, epsilon_min=0.01, epsilon_decay=0.9995,
             method="qlearning", seed=42, rewards_file="rewards.txt",
             repeat_runs=1, avg_out="rewards_mean.txt"):
    """
    Trains the agent for 'epochs', each with 'steps_per_epoch' steps.
    """
    all_runs = []
    for run in range(repeat_runs):
        agent = Agent(params, alpha, epsilon, epsilon_min, epsilon_decay, method, seed + run)
        rng = np.random.default_rng(seed + 10_000 + run)
        rewards_per_epoch = []

        for ep in range(1, epochs + 1):
            s = H
            G = 0.0
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
        agent.save_policy(f"policy_robot_q_run{run+1}.npy")

    np.savetxt(rewards_file, all_runs[-1], fmt="%.6f")
    if repeat_runs > 1:
        mean_curve = np.mean(np.vstack(all_runs), axis=0)
        np.savetxt(avg_out, mean_curve, fmt="%.6f")
        print(f"Saved mean curve to {avg_out}", flush=True)

    print(f"Saved last-run rewards to {rewards_file}", flush=True)
    return all_runs[-1] if repeat_runs == 1 else np.mean(np.vstack(all_runs), axis=0)


def steps_to_threshold(rewards_per_epoch, steps_per_epoch, g_target, eps=0.01, window=10):
    """Calculates the steps required to reach a reward threshold."""
    per_step = rewards_per_epoch / float(steps_per_epoch)
    if len(per_step) < window: return None
    mov = np.convolve(per_step, np.ones(window)/window, mode="valid")
    idxs = np.where(mov >= (g_target - eps))[0]
    if len(idxs) == 0: return None
    reached_epoch = idxs[0] + window
    return reached_epoch * steps_per_epoch


def train_with_metrics(params, epochs, steps_per_epoch, agent_kwargs, seed=42,
                       rewards_file="rewards.txt", q_out="policy_robot_q_run1.npy",
                       dp_value_iteration_fn=None):
    """
    Executes training and returns a dictionary with performance metrics.
    """
    t0 = time.perf_counter()
    _ = train_td(epochs=epochs, steps_per_epoch=steps_per_epoch,
                 params=params, seed=seed, rewards_file=rewards_file, **agent_kwargs)
    t1 = time.perf_counter()
    time_sec = t1 - t0

    r = np.loadtxt(rewards_file, dtype=float)
    Q = np.load(q_out)
    pi_learned = greedy_policy_from_Q(Q)
    g_learned = avg_reward_of_policy(params, pi_learned)

    g_opt = None; steps_eps = None
    if dp_value_iteration_fn is not None:
        (_, _), pi_opt = dp_value_iteration_fn(params)
        g_opt = avg_reward_of_policy(params, pi_opt)
        steps_eps = steps_to_threshold(r, steps_per_epoch, g_opt, eps=0.01, window=10)

    eff_return_per_sec = g_learned / max(time_sec, 1e-9)
    eff_return_per_step = g_learned

    return {
        "time_sec": time_sec,
        "g_learned": g_learned,
        "g_opt": g_opt,
        "gap": (None if g_opt is None else g_opt - g_learned),
        "steps_to_eps_opt": steps_eps,
        "eff_return_per_sec": eff_return_per_sec,
        "eff_return_per_step": eff_return_per_step,
    }