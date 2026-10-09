import json
import time
import zlib
from pathlib import Path
from typing import Dict, List

try:
    import ollama
except ImportError:  # allows --mock runs on machines without Ollama
    ollama = None

from models import Tier, Action, Task, NodeState, NegotiationDecision

TIER_MODELS: Dict[Tier, str] = {
    Tier.MIST: "qwen2.5:0.5b",
    Tier.FOG: "llama3.2:1b",
    Tier.CLOUD: "llama3.1:8b",
}


CHAT_OPTIONS = {"temperature": 0.0, "num_ctx": 1024, "num_predict": 80}
KEEP_ALIVE = "30m"


def build_prompt(tier: Tier, task: Task, state: NodeState) -> str:
    """Compact prompt (about 110 tokens). Shared by the agent and the latency profiler."""
    allowed = (
        '["execute_local", "offload_to_fog"]' if tier == Tier.MIST else
        '["execute_local", "offload_to_cloud", "reject"]' if tier == Tier.FOG else
        '["execute_local", "reject"]'
    )
    return (
        f"You are the {tier.value.upper()} tier resource agent. Decide where to run one task.\n"
        f"Node: cpu_free={state.cpu_free_pct:.0f}%, mem_free={state.mem_free_mb:.0f}MB, "
        f"bandwidth={state.bandwidth_mbps:.0f}Mbps, queue={state.queue_length}\n"
        f"Task: cpu={task.cpu_mflops:.0f}MFLOPS, mem={task.memory_mb:.0f}MB, "
        f"data={task.data_size_mb:.1f}MB, deadline={task.deadline_s:.1f}s\n"
        f"Allowed actions: {allowed}\n"
        'Reply with JSON only: {"action": "<allowed action>", "reason": "<max 10 words>"}'
    )


class SLMAgent:
    """Agent at one tier backed by a local SLM, with schema validation and caching."""

    def __init__(self, tier: Tier, use_cache: bool = True, mock: bool = False):
        self.tier = tier
        self.model_name = TIER_MODELS[tier]
        self.use_cache = use_cache
        self.mock = mock
        self.decision_cache: Dict[str, NegotiationDecision] = {}
        self.total_queries = 0
        self.fallback_count = 0
        self.cache_hits = 0
        self.live_latencies: List[float] = []   # seconds, real model calls only

    # ---------- caching ----------
    def _state_key(self, task: Task, state: NodeState) -> str:
        """Quantised key. Includes every field that appears in the prompt."""
        return (
            f"{self.tier.value}:"
            f"cpu_q{int(state.cpu_free_pct // 20)}:"
            f"mem_q{int(state.mem_free_mb // 256)}:"
            f"q{min(state.queue_length, 6)}:"
            f"t_cpu_q{int(task.cpu_mflops // 500)}:"
            f"t_mem_q{int(task.memory_mb // 150)}:"
            f"d_q{int(task.data_size_mb // 3)}:"
            f"dl_q{int(task.deadline_s // 2)}"
        )

    def save_cache(self, path: Path):
        data = {k: v.model_dump(mode="json") for k, v in self.decision_cache.items()}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load_cache(self, path: Path):
        p = Path(path)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            self.decision_cache = {k: NegotiationDecision.model_validate(v) for k, v in raw.items()}

    # ---------- rule-based decision (format-guard fallback) ----------
    def _rule_decision(self, task: Task, state: NodeState, note: str) -> NegotiationDecision:
        if self.tier == Tier.MIST:
            if task.cpu_mflops > 800 or task.memory_mb > 300 or state.cpu_free_pct < 35:
                return NegotiationDecision(action=Action.OFFLOAD_TO_FOG, target_tier=Tier.FOG,
                                           reason=f"[GUARD FALLBACK] High resource demand: {note}", is_fallback=True)
            return NegotiationDecision(action=Action.EXECUTE_LOCAL, target_tier=Tier.MIST,
                                       reason=f"[GUARD FALLBACK] Light task within Mist capacity: {note}", is_fallback=True)
        if self.tier == Tier.FOG:
            if state.queue_length >= 4 or state.cpu_free_pct < 25 or task.cpu_mflops > 2500:
                return NegotiationDecision(action=Action.OFFLOAD_TO_CLOUD, target_tier=Tier.CLOUD,
                                           reason=f"[GUARD FALLBACK] Fog overloaded, forwarding to Cloud: {note}", is_fallback=True)
            return NegotiationDecision(action=Action.EXECUTE_LOCAL, target_tier=Tier.FOG,
                                       reason=f"[GUARD FALLBACK] Fog has sufficient capacity: {note}", is_fallback=True)
        if state.cpu_free_pct < 5 and state.queue_length > 20:
            return NegotiationDecision(action=Action.REJECT,
                                       reason=f"[GUARD FALLBACK] Cloud extreme saturation: {note}", is_fallback=True)
        return NegotiationDecision(action=Action.EXECUTE_LOCAL, target_tier=Tier.CLOUD,
                                   reason=f"[GUARD FALLBACK] Cloud high-performance execution: {note}", is_fallback=True)

    def _rule_based_fallback(self, task: Task, state: NodeState, note: str) -> NegotiationDecision:
        self.fallback_count += 1
        return self._rule_decision(task, state, note)

    # ---------- mock agent (offline testing only; NOT a real SLM) ----------
    def _mock_decide(self, task: Task, state: NodeState, key: str) -> NegotiationDecision:
        base = self._rule_decision(task, state, "mock").model_copy()
        base.is_fallback = False
        base.reason = "[MOCK AGENT]"
        if zlib.crc32(key.encode()) % 100 < 15:   # deterministic ~15% "bad" decisions
            if base.action == Action.EXECUTE_LOCAL:
                base.action = {Tier.MIST: Action.OFFLOAD_TO_FOG, Tier.FOG: Action.OFFLOAD_TO_CLOUD,
                               Tier.CLOUD: Action.REJECT}[self.tier]
                base.target_tier = None
            else:
                base.action, base.target_tier = Action.EXECUTE_LOCAL, self.tier
        return base

    # ---------- main entry ----------
    def decide(self, task: Task, state: NodeState, force_live_query: bool = False) -> NegotiationDecision:
        self.total_queries += 1
        cache_key = self._state_key(task, state)

        if self.use_cache and not force_live_query and cache_key in self.decision_cache:
            self.cache_hits += 1
            cached = self.decision_cache[cache_key].model_copy()
            cached.decision_latency_s = 0.001
            return cached

        if self.mock:
            decision = self._mock_decide(task, state, cache_key)
            if self.use_cache:
                self.decision_cache[cache_key] = decision
            return decision

        prompt = build_prompt(self.tier, task, state)

        cacheable = True
        start_t = time.perf_counter()
        try:
            resp = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options=CHAT_OPTIONS,
                keep_alive=KEEP_ALIVE,
            )
            elapsed = time.perf_counter() - start_t
            self.live_latencies.append(elapsed)
            content = resp.get("message", {}).get("content", "").strip()

            data = json.loads(content)
            action_str = str(data.get("action", "")).lower().replace("-", "_").replace(" ", "_")
            reason = data.get("reason", "SLM evaluated")

            action_map = {
                "execute_local": Action.EXECUTE_LOCAL,
                "local": Action.EXECUTE_LOCAL,
                "offload": Action.OFFLOAD_TO_FOG if self.tier == Tier.MIST else Action.OFFLOAD_TO_CLOUD,
                "offload_to_fog": Action.OFFLOAD_TO_FOG,
                "offload_to_cloud": Action.OFFLOAD_TO_CLOUD,
                "reject": Action.REJECT,
            }
            if action_str not in action_map:
                decision = self._rule_based_fallback(task, state, f"Unrecognized action '{action_str}'")
                decision.decision_latency_s = elapsed
            else:
                act = action_map[action_str]
                tgt = {Action.OFFLOAD_TO_FOG: Tier.FOG, Action.OFFLOAD_TO_CLOUD: Tier.CLOUD,
                       Action.EXECUTE_LOCAL: self.tier}.get(act)
                decision = NegotiationDecision(
                    action=act, target_tier=tgt, reason=f"[SLM {self.model_name}] {reason}",
                    is_fallback=False, decision_latency_s=elapsed,
                )
        except Exception as e:
            elapsed = time.perf_counter() - start_t
            decision = self._rule_based_fallback(task, state, str(e))
            decision.decision_latency_s = elapsed
            cacheable = False   # never cache transient failures (e.g. Ollama down)

        if self.use_cache and cacheable:
            self.decision_cache[cache_key] = decision
        return decision