# -*- coding: utf-8 -*-
"""
Program entry point.
Defines simulation parameters and orchestrates training and analysis.
"""
import os
import numpy as np
from robot_mdp import Params, build_P_R_tables, plot_sa_to_sprime_heatmap, plot_expected_reward_heatmap
from training import train_td, train_with_metrics
from analysis import plot_epoch_metrics, plot_policy_heatmap
from dp_solvers import value_iteration

# --- Global entry point parameters ---
STEPS_PER_EPOCH = 2000
EPOCHS = 500
REPEAT_RUNS = 5

if __name__ == "__main__":
    # --- 1. Define problem parameters ---
    params = Params(alpha_p=0.5, beta_p=0.5, r_search=1.0, r_wait=0.5, penalty=-3.0, gamma=0.5)

    # --- 2. Example execution to collect performance metrics ---
    print("--- Executing training with metrics ---")
    agent_kwargs = dict(alpha=0.2, epsilon=0.3, epsilon_min=0.01, epsilon_decay=0.99, method="qlearning")
    metrics = train_with_metrics(
        params,
        epochs=60,
        steps_per_epoch=STEPS_PER_EPOCH,
        agent_kwargs=agent_kwargs,
        seed=42,
        dp_value_iteration_fn=value_iteration
    )
    print("\n--- Performance Metrics ---")
    print(f"Execution time (s): {metrics['time_sec']:.2f}")
    print(f"Learned average reward (g): {metrics['g_learned']:.3f}")
    if metrics['g_opt'] is not None:
        print(f"Optimal average reward (g*): {metrics['g_opt']:.3f}")
        print(f"Gap (g* - g): {metrics['gap']:.3f}")
        if metrics['steps_to_eps_opt'] is not None:
            print(f"Steps to converge (within 1% of g*): {metrics['steps_to_eps_opt']}")
        else:
            print("Convergence threshold was not reached.")
    print("-" * 30)

    # --- 3. Run for plotting (multiple runs) ---
    print("\n--- Executing full training for plotting ---")
    train_td(
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

    # --- 4. Plot results ---
    print("\n--- Generating plots ---")
    plot_epoch_metrics(
        rewards_path=f"rewards_mean.txt" if REPEAT_RUNS > 1 else "rewards.txt",
        steps_per_epoch=STEPS_PER_EPOCH,
        Q_path=f"policy_robot_q_run{REPEAT_RUNS}.npy",
        params=params,
        window=10,
        title_suffix=f"(αp={params.alpha_p}, βp={params.beta_p}, γ={params.gamma})"
    )

    plot_policy_heatmap(Q_path=f"policy_robot_q_run{REPEAT_RUNS}.npy", epsilon_eval=0.05)

    # --- 5. Environment visualization ---
    print("\n--- Generating environment plots (P and R tables) ---")
    P_sa, R_sa = build_P_R_tables(params)
    plot_sa_to_sprime_heatmap(P_sa)
    plot_expected_reward_heatmap(R_sa)