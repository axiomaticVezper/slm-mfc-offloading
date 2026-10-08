import os
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_evaluation_plots(summary_df: pd.DataFrame, detailed_df: pd.DataFrame, output_dir: str):
    """Generates comparative visualization figures for the evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Deadline Satisfaction Rate (%)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    bars = ax.bar(summary_df["policy"], summary_df["deadline_hit_rate_pct"], color=colors, width=0.55)
    ax.set_title("Task Deadline Satisfaction Rate (%) by Offloading Policy", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Deadline Hit Rate (%)", fontsize=11)
    ax.set_ylim(0, 105)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "deadline_satisfaction.png"), dpi=200)
    plt.close()

    # 2. Average Turnaround Time (s)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary_df["policy"], summary_df["avg_turnaround_s"], color=colors, width=0.55)
    ax.set_title("Average Task Turnaround Latency (s)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Turnaround Time (seconds)", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "turnaround_latency.png"), dpi=200)
    plt.close()

    # 3. Task Distribution by Tier (%)
    tier_dist = detailed_df.groupby(["policy", "executed_tier"]).size().unstack(fill_value=0)
    # Ensure all columns exist
    for col in ["mist", "fog", "cloud"]:
        if col not in tier_dist.columns:
            tier_dist[col] = 0
    tier_dist_pct = tier_dist.div(tier_dist.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    tier_dist_pct[["mist", "fog", "cloud"]].plot(
        kind="bar",
        stacked=True,
        color=["#1abc9c", "#3498db", "#9b59b6"],
        ax=ax,
        width=0.55
    )
    ax.set_title("Workload Execution Tier Distribution (%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Percentage of Tasks (%)", fontsize=11)
    ax.set_xlabel("Offloading Policy", fontsize=11)
    ax.legend(["Mist (IoT)", "Fog (Edge Gateway)", "Cloud (Data Center)"], loc="upper right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tier_distribution.png"), dpi=200)
    plt.close()

    # 4. Total Energy Consumption (Joules)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary_df["policy"], summary_df["avg_energy_joules"], color=["#e67e22", "#f1c40f", "#2980b9", "#27ae60"], width=0.55)
    ax.set_title("Average Energy Consumption per Task (Joules)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Energy (Joules)", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f} J",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_consumption.png"), dpi=200)
    plt.close()
