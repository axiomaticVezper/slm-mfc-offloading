"""Resource-feasibility guard (the project's novelty).

SLM agents may propose placements that look plausible in text but are physically
infeasible (not enough free memory, or the task cannot finish before its deadline
given the queue). The guard is a deterministic, rule-based validator that checks every
proposal against the node's REAL state before it is accepted, and overrides it if needed.
It is separate from the format (JSON/schema) fallback inside SLMAgent.
"""
from models import Tier, Action, Task, NodeState, NegotiationDecision

# (capacity MFLOPS/s, cores, memory MB) -- must match the ComputeNode configuration
NODE_SPECS = {
    Tier.MIST: (1200.0, 1, 512.0),
    Tier.FOG: (7500.0, 2, 4096.0),
    Tier.CLOUD: (35000.0, 8, 32768.0),
}


class ResourceGuard:
    def __init__(self, specs=None):
        self.specs = specs or NODE_SPECS
        self.override_count = 0

    def _local_ok(self, tier: Tier, task: Task, state: NodeState, remaining_s: float):
        cap, cores, _ = self.specs[tier]
        if task.memory_mb > state.mem_free_mb:
            return False, f"memory {task.memory_mb:.0f}MB > free {state.mem_free_mb:.0f}MB"
        compute = task.cpu_mflops / cap
        ahead = max(0, state.queue_length - cores + 1)      # tasks that must finish first
        est_finish = ahead * compute / cores + compute      # heuristic queue-wait estimate
        if est_finish > remaining_s:
            return False, f"predicted finish {est_finish:.2f}s > remaining {remaining_s:.2f}s"
        return True, ""

    def _override(self, dec, action, target, reason):
        self.override_count += 1
        return NegotiationDecision(
            action=action, target_tier=target, reason=f"[RESOURCE GUARD] {reason}",
            is_fallback=dec.is_fallback, guard_overridden=True,
            fault_injected=dec.fault_injected, decided_by=dec.decided_by,
            decision_latency_s=dec.decision_latency_s,
        )

    def check(self, tier: Tier, task: Task, state: NodeState,
              dec: NegotiationDecision, remaining_s: float) -> NegotiationDecision:
        # 1) A "run locally" proposal must be physically feasible
        if dec.action == Action.EXECUTE_LOCAL:
            ok, why = self._local_ok(tier, task, state, remaining_s)
            if ok:
                return dec
            if tier == Tier.MIST:
                return self._override(dec, Action.OFFLOAD_TO_FOG, Tier.FOG, f"Mist cannot run locally ({why})")
            if tier == Tier.FOG:
                return self._override(dec, Action.OFFLOAD_TO_CLOUD, Tier.CLOUD, f"Fog cannot run locally ({why})")
            # Cloud: no higher tier. Only memory infeasibility justifies a reject.
            if task.memory_mb > state.mem_free_mb:
                return self._override(dec, Action.REJECT, None, f"Cloud cannot run task ({why})")
            return dec

        # 2) A "reject" proposal is overridden when a feasible tier still exists
        if dec.action == Action.REJECT:
            if tier == Tier.FOG and task.memory_mb <= self.specs[Tier.CLOUD][2]:
                return self._override(dec, Action.OFFLOAD_TO_CLOUD, Tier.CLOUD, "Reject unjustified; Cloud can host task")
            if tier == Tier.CLOUD:
                ok, _ = self._local_ok(tier, task, state, remaining_s)
                if ok:
                    return self._override(dec, Action.EXECUTE_LOCAL, Tier.CLOUD, "Reject unjustified; Cloud has capacity")
        return dec