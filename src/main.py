r"""Benchmark driver (upgraded).

Usage (from the project root, venv active):
    python src\main.py --seeds 3            # real Ollama run
    python src\main.py --mock --seeds 2     # offline logic test, no Ollama (NOT for the report)
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models import Tier, Task
from simulation import MFCSimulator
from llm_agent import SLMAgent, TIER_MODELS
from plotter import generate_evaluation_plots

ROOT = Path(__file__).resolve().parent.parent

# mean inter-arrival time (s) of the exponential arrival process
SCENARIOS = {"normal": 0.8, "heavy_load": 0.3}

CONFIGS = [
    dict(name="local_only", policy="local_only"),
    dict(name="greedy", policy="greedy"),
    dict(name="static_threshold", policy="static_threshold"),
    dict(name="always_fog", policy="always_fog"),
    dict(name="slm_noguard", policy="slm_negotiation", guard=False, fault=0.0),
    dict(name="slm_guard", policy="slm_negotiation", guard=True, fault=0.0),
    dict(name="slm_noguard_faulty", policy="slm_negotiation", guard=False, fault=0.2),
    dict(name="slm_guard_faulty", policy="slm_negotiation", guard=True, fault=0.2),
]
CHARGED = dict(name="slm_guard_charged", policy="slm_negotiation", guard=True, fault=0.0)


def generate_workload(num_tasks, seed, mean_ia):
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    tasks, t = [], 0.0
    for i in range(num_tasks):
        t += float(rng.exponential(scale=mean_ia))
        p = rnd.random()
        if p < 0.40:
            tasks.append(Task(task_id=f"task_light_{i:03d}", cpu_mflops=rnd.uniform(300, 750),
                              memory_mb=rnd.uniform(50, 150), data_size_mb=rnd.uniform(0.1, 0.8),
                              deadline_s=rnd.uniform(1.8, 3.0), arrival_time=t))
        elif p < 0.80:
            tasks.append(Task(task_id=f"task_med_{i:03d}", cpu_mflops=rnd.uniform(1500, 4200),
                              memory_mb=rnd.uniform(250, 750), data_size_mb=rnd.uniform(2.0, 7.0),
                              deadline_s=rnd.uniform(3.5, 6.0), arrival_time=t))
        else:
            tasks.append(Task(task_id=f"task_heavy_{i:03d}", cpu_mflops=rnd.uniform(9000, 22000),
                              memory_mb=rnd.uniform(1200, 3500), data_size_mb=rnd.uniform(10.0, 25.0),
                              deadline_s=rnd.uniform(7.0, 15.0), arrival_time=t))
    return tasks


def compute_metrics(df):
    n = len(df)
    ex = df[df.executed_tier.notna()]
    share = lambda tier: float((ex.executed_tier == tier).sum()) / n * 100
    has = len(ex) > 0
    return dict(
        n_tasks=n,
        deadline_hit_rate_pct=df.met_deadline.mean() * 100,
        avg_turnaround_s=ex.total_turnaround.mean() if has else np.nan,
        p95_turnaround_s=ex.total_turnaround.quantile(0.95) if has else np.nan,
        avg_queue_delay_s=ex.queue_delay.mean() if has else np.nan,
        avg_energy_j=ex.energy_joules.mean() if has else np.nan,
        rejected_pct=(n - len(ex)) / n * 100,
        mem_violation_pct=df.mem_violation.mean() * 100,
        overloaded_pct=float((ex.queue_delay > 1.0).sum()) / n * 100,   # queue wait > 1 s
        fallback_pct=df.fallback_used.mean() * 100,
        guard_override_pct=df.guard_override.mean() * 100,
        mist_pct=share("mist"), fog_pct=share("fog"), cloud_pct=share("cloud"),
        mean_decision_latency_s=df.decision_latency_s.mean(),
    )


def check_ollama():
    import ollama
    for tier, model in TIER_MODELS.items():
        try:
            ollama.chat(model=model, messages=[{"role": "user", "content": "Reply with the word ok"}])
            print(f"  [ok] {tier.value}: {model}")
        except Exception as e:
            print(f"  [FAIL] {tier.value}: {model} -> {e}")
            print("Fix: start Ollama and run `ollama pull " + model + "`. Aborting so no results are polluted by fallbacks.")
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--num-tasks", type=int, default=60)
    ap.add_argument("--mock", action="store_true", help="offline logic test with a fake agent")
    ap.add_argument("--fresh-cache", action="store_true", help="ignore saved SLM decision cache")
    args = ap.parse_args()

    suffix = "_mock" if args.mock else ""
    results_dir, figures_dir = ROOT / f"results{suffix}", ROOT / f"figures{suffix}"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    seeds = [101 + i for i in range(args.seeds)]

    print("=" * 70)
    print("Multi-Agent SLM Task Offloading: guarded negotiation benchmark" + (" [MOCK AGENTS]" if args.mock else ""))
    print("=" * 70)

    if not args.mock:
        print("Checking Ollama models (this also loads them into memory)...")
        check_ollama()

    agents = {t: SLMAgent(t, use_cache=True, mock=args.mock) for t in Tier}
    if not args.mock and not args.fresh_cache:
        for t, a in agents.items():
            a.load_cache(results_dir / f"slm_cache_{t.value}.json")

    def save_caches():
        if not args.mock:
            for t, a in agents.items():
                a.save_cache(results_dir / f"slm_cache_{t.value}.json")

    run_rows, task_frames = [], []

    def run_one(scenario, seed, cfg, tasks, inference_cost=None):
        sim = MFCSimulator(policy=cfg["policy"], agents=agents, use_guard=cfg.get("guard", False),
                           fault_rate=cfg.get("fault", 0.0), seed=seed, inference_cost=inference_cost)
        res = sim.run(tasks)
        df = pd.DataFrame([r.model_dump(mode="json") for r in res])
        df["scenario"], df["seed"], df["config"] = scenario, seed, cfg["name"]
        task_frames.append(df)
        run_rows.append(dict(scenario=scenario, seed=seed, config=cfg["name"], **compute_metrics(df)))

    t0 = time.perf_counter()
    for scenario, ia in SCENARIOS.items():
        for seed in seeds:
            tasks = generate_workload(args.num_tasks, seed, ia)
            for cfg in CONFIGS:
                run_one(scenario, seed, cfg, tasks)
            save_caches()
            print(f"[{time.perf_counter() - t0:7.1f}s] scenario={scenario} seed={seed} done "
                  f"(live SLM calls so far: {sum(len(a.live_latencies) for a in agents.values())})")

    # Phase 2: charge measured per-tier inference time to simulated time
    cost = {t: (float(np.mean(a.live_latencies)) if a.live_latencies else 0.0) for t, a in agents.items()}
    print("Mean live inference latency per tier (s):", {t.value: round(v, 2) for t, v in cost.items()})
    for scenario, ia in SCENARIOS.items():
        for seed in seeds:
            run_one(scenario, seed, CHARGED, generate_workload(args.num_tasks, seed, ia), inference_cost=cost)

    runs_df = pd.DataFrame(run_rows)
    tasks_df = pd.concat(task_frames, ignore_index=True)

    summary = runs_df.drop(columns=["seed"]).groupby(["scenario", "config"]).agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().round(3)

    lat = pd.DataFrame([dict(
        tier=t.value, model=a.model_name, total_queries=a.total_queries, cache_hits=a.cache_hits,
        live_calls=len(a.live_latencies), format_fallbacks=a.fallback_count,
        mean_live_latency_s=float(np.mean(a.live_latencies)) if a.live_latencies else np.nan,
        p95_live_latency_s=float(np.percentile(a.live_latencies, 95)) if a.live_latencies else np.nan,
    ) for t, a in agents.items()])

    runs_df.round(4).to_csv(results_dir / "run_metrics.csv", index=False)
    summary.to_csv(results_dir / "summary_metrics.csv", index=False)
    tasks_df.to_csv(results_dir / "detailed_tasks.csv", index=False)
    lat.to_csv(results_dir / "slm_latency_stats.csv", index=False)
    generate_evaluation_plots(runs_df, figures_dir)

    show = ["scenario", "config", "deadline_hit_rate_pct_mean", "avg_turnaround_s_mean", "avg_energy_j_mean",
            "mem_violation_pct_mean", "overloaded_pct_mean", "guard_override_pct_mean"]
    print("\n" + summary[show].to_string(index=False))
    print("\nSLM latency / cache stats:\n" + lat.to_string(index=False))
    print(f"\nSaved to: {results_dir}  and  {figures_dir}")


if __name__ == "__main__":
    main()