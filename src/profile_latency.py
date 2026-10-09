r"""Latency profiler: separates model LOAD, PROMPT-PROCESSING and GENERATION time.

Run from the project root with the venv active:
    python src\profile_latency.py            # Mist + Fog tests
    python src\profile_latency.py --cloud    # also tests the 8B model (slow, loads ~5 GB)
Uses the same prompt, options and keep-alive as the real agents (imported from llm_agent.py).
"""
import argparse
import time

import ollama

from llm_agent import TIER_MODELS, CHAT_OPTIONS, KEEP_ALIVE, build_prompt
from models import Tier, Task, NodeState

TASK = Task(task_id="t", cpu_mflops=2800, memory_mb=420, data_size_mb=4.5, deadline_s=4.5)
STATE = {
    Tier.MIST: NodeState(tier=Tier.MIST, cpu_free_pct=100, mem_free_mb=512, bandwidth_mbps=30, queue_length=0),
    Tier.FOG: NodeState(tier=Tier.FOG, cpu_free_pct=100, mem_free_mb=4096, bandwidth_mbps=250, queue_length=0),
    Tier.CLOUD: NodeState(tier=Tier.CLOUD, cpu_free_pct=100, mem_free_mb=32768, bandwidth_mbps=500, queue_length=0),
}


def call(label, tier):
    model = TIER_MODELS[tier]
    t = time.perf_counter()
    r = ollama.chat(model=model, messages=[{"role": "user", "content": build_prompt(tier, TASK, STATE[tier])}],
                    format="json", options=CHAT_OPTIONS, keep_alive=KEEP_ALIVE)
    wall = time.perf_counter() - t
    ns = 1e9
    print(f"{label:<20} {model:<13} wall={wall:6.2f}s load={r.get('load_duration', 0) / ns:5.2f}s "
          f"prompt={r.get('prompt_eval_duration', 0) / ns:5.2f}s ({r.get('prompt_eval_count', 0):3d} tok) "
          f"gen={r.get('eval_duration', 0) / ns:5.2f}s ({r.get('eval_count', 0):3d} tok)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", action="store_true")
    args = ap.parse_args()

    print("A) Mist model, 5 consecutive calls (first may include load)")
    for i in range(5):
        call(f"mist #{i + 1}", Tier.MIST)

    print("\nB) Alternating Mist / Fog, 4 pairs (reloads show up as large 'load')")
    for i in range(4):
        call(f"mist alt #{i + 1}", Tier.MIST)
        call(f"fog  alt #{i + 1}", Tier.FOG)

    if args.cloud:
        print("\nC) 8B Cloud model, then small models again (does the 8B evict them?)")
        call("cloud #1", Tier.CLOUD)
        call("cloud #2", Tier.CLOUD)
        call("mist after cloud #1", Tier.MIST)
        call("mist after cloud #2", Tier.MIST)
        call("fog  after cloud", Tier.FOG)
        call("fog  again", Tier.FOG)


if __name__ == "__main__":
    main()