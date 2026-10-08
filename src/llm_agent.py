import json
import time
from typing import Dict, Tuple, Optional
import ollama
from models import Tier, Action, Task, NodeState, NegotiationDecision

# Tier to Model Mapping
TIER_MODELS: Dict[Tier, str] = {
    Tier.MIST: "qwen2.5:0.5b",
    Tier.FOG: "llama3.2:1b",
    Tier.CLOUD: "llama3.1:8b",
}

class SLMAgent:
    """Agent running at a specific tier using a localized Small Language Model."""

    def __init__(self, tier: Tier, use_cache: bool = True):
        self.tier = tier
        self.model_name = TIER_MODELS[tier]
        self.use_cache = use_cache
        self.decision_cache: Dict[str, NegotiationDecision] = {}
        self.total_queries = 0
        self.fallback_count = 0
        self.cache_hits = 0

    def _state_key(self, task: Task, state: NodeState) -> str:
        """Quantizes state to allow smart caching for high-speed simulation."""
        return (
            f"{self.tier.value}:"
            f"cpu_q{int(state.cpu_free_pct // 20)}:"
            f"mem_q{int(state.mem_free_mb // 256)}:"
            f"t_cpu_q{int(task.cpu_mflops // 500)}:"
            f"t_mem_q{int(task.memory_mb // 150)}:"
            f"dl_q{int(task.deadline_s // 2)}"
        )

    def _rule_based_fallback(self, task: Task, state: NodeState, reason_note: str) -> NegotiationDecision:
        """Guarded deterministic fallback executed if SLM fails, malforms JSON, or times out."""
        self.fallback_count += 1
        
        if self.tier == Tier.MIST:
            # Mist: constrained node. If task needs > 50% CPU or memory, offload to Fog
            if task.cpu_mflops > 800 or task.memory_mb > 300 or state.cpu_free_pct < 35:
                return NegotiationDecision(
                    action=Action.OFFLOAD_TO_FOG,
                    target_tier=Tier.FOG,
                    reason=f"[GUARD FALLBACK] High resource demand: {reason_note}",
                    is_fallback=True
                )
            return NegotiationDecision(
                action=Action.EXECUTE_LOCAL,
                target_tier=Tier.MIST,
                reason=f"[GUARD FALLBACK] Light task within Mist capacity: {reason_note}",
                is_fallback=True
            )

        elif self.tier == Tier.FOG:
            # Fog: mid-tier gateway. If queue > 4 or CPU < 25%, forward to Cloud
            if state.queue_length >= 4 or state.cpu_free_pct < 25 or task.cpu_mflops > 2500:
                return NegotiationDecision(
                    action=Action.OFFLOAD_TO_CLOUD,
                    target_tier=Tier.CLOUD,
                    reason=f"[GUARD FALLBACK] Fog overloaded, forwarding to Cloud: {reason_note}",
                    is_fallback=True
                )
            return NegotiationDecision(
                action=Action.EXECUTE_LOCAL,
                target_tier=Tier.FOG,
                reason=f"[GUARD FALLBACK] Fog has sufficient capacity: {reason_note}",
                is_fallback=True
            )

        else: # Cloud
            if state.cpu_free_pct < 5 and state.queue_length > 20:
                return NegotiationDecision(
                    action=Action.REJECT,
                    reason=f"[GUARD FALLBACK] Cloud extreme saturation: {reason_note}",
                    is_fallback=True
                )
            return NegotiationDecision(
                action=Action.EXECUTE_LOCAL,
                target_tier=Tier.CLOUD,
                reason=f"[GUARD FALLBACK] Cloud high-performance execution: {reason_note}",
                is_fallback=True
            )

    def decide(self, task: Task, state: NodeState, force_live_query: bool = False) -> NegotiationDecision:
        """Queries the tier SLM with strict schema validation & fallback protection."""
        self.total_queries += 1
        cache_key = self._state_key(task, state)

        if self.use_cache and not force_live_query and cache_key in self.decision_cache:
            self.cache_hits += 1
            cached = self.decision_cache[cache_key].model_copy()
            cached.decision_latency_s = 0.001
            return cached

        # Prepare context prompt
        actions_allowed = (
            '["execute_local", "offload_to_fog"]' if self.tier == Tier.MIST else
            '["execute_local", "offload_to_cloud", "reject"]' if self.tier == Tier.FOG else
            '["execute_local", "reject"]'
        )

        prompt = f"""You are the autonomous resource allocation agent for the {self.tier.value.upper()} tier.
Node Status:
- CPU Free: {state.cpu_free_pct:.1f}%
- Memory Free: {state.mem_free_mb:.0f} MB
- Bandwidth: {state.bandwidth_mbps:.1f} Mbps
- Current Queue: {state.queue_length} tasks

Incoming Task:
- ID: {task.task_id}
- CPU Demand: {task.cpu_mflops:.0f} MFLOPS
- Memory Demand: {task.memory_mb:.0f} MB
- Data Size: {task.data_size_mb:.1f} MB
- Latency Deadline: {task.deadline_s:.1f} seconds

Allowed actions: {actions_allowed}.
Decide whether to execute locally, offload to next tier, or reject.
Respond ONLY with a valid JSON object matching this schema:
{{
  "action": "<one of allowed actions>",
  "target_tier": "<mist|fog|cloud or null>",
  "reason": "<one sentence justification>"
}}"""

        start_t = time.perf_counter()
        try:
            resp = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0}
            )
            elapsed = time.perf_counter() - start_t
            content = resp.get("message", {}).get("content", "").strip()

            data = json.loads(content)
            action_str = data.get("action", "").lower().replace("-", "_").replace(" ", "_")
            target_str = data.get("target_tier")
            reason = data.get("reason", "SLM evaluated")

            # Validate action
            action_map = {
                "execute_local": Action.EXECUTE_LOCAL,
                "local": Action.EXECUTE_LOCAL,
                "offload": Action.OFFLOAD_TO_FOG if self.tier == Tier.MIST else Action.OFFLOAD_TO_CLOUD,
                "offload_to_fog": Action.OFFLOAD_TO_FOG,
                "offload_to_cloud": Action.OFFLOAD_TO_CLOUD,
                "reject": Action.REJECT
            }

            if action_str not in action_map:
                decision = self._rule_based_fallback(task, state, f"Unrecognized action '{action_str}'")
            else:
                act = action_map[action_str]
                # Canonical target tier
                tgt = None
                if act == Action.OFFLOAD_TO_FOG:
                    tgt = Tier.FOG
                elif act == Action.OFFLOAD_TO_CLOUD:
                    tgt = Tier.CLOUD
                elif act == Action.EXECUTE_LOCAL:
                    tgt = self.tier

                decision = NegotiationDecision(
                    action=act,
                    target_tier=tgt,
                    reason=f"[SLM {self.model_name}] {reason}",
                    is_fallback=False,
                    decision_latency_s=elapsed
                )

        except Exception as e:
            elapsed = time.perf_counter() - start_t
            decision = self._rule_based_fallback(task, state, str(e))
            decision.decision_latency_s = elapsed

        if self.use_cache:
            self.decision_cache[cache_key] = decision

        return decision
