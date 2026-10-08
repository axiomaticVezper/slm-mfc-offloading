import os
import random
import time
import pandas as pd
import numpy as np
from typing import List

from models import Tier, Task, NodeState
from simulation import MFCSimulator
from llm_agent import SLMAgent
from plotter import generate_evaluation_plots

def generate_workload(num_tasks: int = 40, seed: int = 42) -> List[Task]:
    """Generates heterogeneous IoT/Edge/Cloud workload."""
    random.seed(seed)
    np.random.seed(seed)
    tasks = []
    current_time = 0.0

    for i in range(num_tasks):
        # Poisson inter-arrival with mean rate 0.8s
        inter_arrival = float(np.random.exponential(scale=0.8))
        current_time += inter_arrival
        
        # 40% Light, 40% Medium, 20% Heavy
        rand_profile = random.random()
        if rand_profile < 0.40:
            # Light IoT task
            t = Task(
                task_id=f"task_light_{i:02d}",
                cpu_mflops=float(random.uniform(300, 750)),
                memory_mb=float(random.uniform(50, 150)),
                data_size_mb=float(random.uniform(0.1, 0.8)),
                deadline_s=float(random.uniform(1.8, 3.0)),
                arrival_time=current_time
            )
        elif rand_profile < 0.80:
            # Medium Edge task
            t = Task(
                task_id=f"task_med_{i:02d}",
                cpu_mflops=float(random.uniform(1500, 4200)),
                memory_mb=float(random.uniform(250, 750)),
                data_size_mb=float(random.uniform(2.0, 7.0)),
                deadline_s=float(random.uniform(3.5, 6.0)),
                arrival_time=current_time
            )
        else:
            # Heavy Cloud task
            t = Task(
                task_id=f"task_heavy_{i:02d}",
                cpu_mflops=float(random.uniform(9000, 22000)),
                memory_mb=float(random.uniform(1200, 3500)),
                data_size_mb=float(random.uniform(10.0, 25.0)),
                deadline_s=float(random.uniform(7.0, 15.0)),
                arrival_time=current_time
            )
        tasks.append(t)
    return tasks

def main():
    print("=" * 70)
    print("Multi-Agent SLM-Based Adaptive Task Offloading & Negotiation")
    print("=" * 70)

    workload = generate_workload(num_tasks=40, seed=123)
    print(f"Generated {len(workload)} heterogeneous tasks.")

    # Initialize shared SLM agents (Mist: qwen2.5:0.5b, Fog: llama3.2:1b, Cloud: llama3.1:8b)
    print("\nInitializing tier SLM agents...")
    agents = {
        Tier.MIST: SLMAgent(Tier.MIST, use_cache=True),
        Tier.FOG: SLMAgent(Tier.FOG, use_cache=True),
        Tier.CLOUD: SLMAgent(Tier.CLOUD, use_cache=True)
    }

    # Pre-warm Mist agent with a live query to verify Ollama integration
    print("Running initial live SLM verification query with Mist agent...")
    sample_state = NodeState(
        tier=Tier.MIST,
        cpu_free_pct=60.0,
        mem_free_mb=350.0,
        bandwidth_mbps=30.0,
        queue_length=1
    )
    init_dec = agents[Tier.MIST].decide(workload[0], sample_state, force_live_query=True)
    print(f"Mist initial decision: {init_dec.action.value} (target={init_dec.target_tier}, latency={init_dec.decision_latency_s:.2f}s)")
    print(f"Reason: {init_dec.reason}")

    policies = ["local_only", "greedy", "static_threshold", "slm_negotiation"]
    all_results = []
    summary_rows = []

    for pol in policies:
        print(f"\n---> Running Simulation Policy: [{pol}]")
        sim = MFCSimulator(policy=pol, agents=agents)
        sim.env.process(sim.run_workload(workload))
        start_wall = time.perf_counter()
        sim.env.run()
        wall_time = time.perf_counter() - start_wall

        df_pol = pd.DataFrame([r.model_dump() for r in sim.results])
        all_results.append(df_pol)

        total_tasks = len(df_pol)
        met_count = int(df_pol["met_deadline"].sum())
        hit_rate = (met_count / total_tasks) * 100.0
        avg_turnaround = float(df_pol["total_turnaround"].mean())
        avg_energy = float(df_pol["energy_joules"].mean())
        avg_queue = float(df_pol["queue_delay"].mean())
        fallback_rate = (float(df_pol["fallback_used"].sum()) / total_tasks) * 100.0

        summary_rows.append({
            "policy": pol,
            "total_tasks": total_tasks,
            "deadline_hit_rate_pct": round(hit_rate, 1),
            "avg_turnaround_s": round(avg_turnaround, 3),
            "avg_queue_delay_s": round(avg_queue, 3),
            "avg_energy_joules": round(avg_energy, 1),
            "fallback_rate_pct": round(fallback_rate, 1),
            "sim_wall_time_s": round(wall_time, 2)
        })
        print(f"Policy [{pol}] Completed | Hit Rate: {hit_rate:.1f}% | Avg Turnaround: {avg_turnaround:.2f}s | Avg Energy: {avg_energy:.1f}J")

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.concat(all_results, ignore_index=True)

    # Save results
    results_dir = os.path.expanduser(r"~\DA2_MFC\results")
    figures_dir = os.path.expanduser(r"~\DA2_MFC\figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    summary_df.to_csv(os.path.join(results_dir, "summary_metrics.csv"), index=False)
    detailed_df.to_csv(os.path.join(results_dir, "detailed_tasks.csv"), index=False)

    print("\nGenerating evaluation plots in figures/...")
    generate_evaluation_plots(summary_df, detailed_df, figures_dir)

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"\nArtifacts saved:")
    print(f"- Summary CSV: {os.path.join(results_dir, 'summary_metrics.csv')}")
    print(f"- Figures: {figures_dir}")

if __name__ == "__main__":
    main()
