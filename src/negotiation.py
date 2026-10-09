import random
from typing import Dict, List, Optional, Tuple

from models import Tier, Action, Task, NodeState, NegotiationDecision
from network import NetworkChannel
from llm_agent import SLMAgent
from guard import ResourceGuard

ALLOWED = {
    Tier.MIST: {Action.EXECUTE_LOCAL, Action.OFFLOAD_TO_FOG},
    Tier.FOG: {Action.EXECUTE_LOCAL, Action.OFFLOAD_TO_CLOUD, Action.REJECT},
    Tier.CLOUD: {Action.EXECUTE_LOCAL, Action.REJECT},
}
_TARGET = {Action.OFFLOAD_TO_FOG: Tier.FOG, Action.OFFLOAD_TO_CLOUD: Tier.CLOUD}


class NegotiationProtocol:
    """Hierarchical Mist -> Fog -> Cloud negotiation with optional resource guard
    and optional fault injection (stress test of agent unreliability)."""

    def __init__(self, agents: Dict[Tier, SLMAgent], guard: Optional[ResourceGuard] = None,
                 fault_rate: float = 0.0, seed: int = 0):
        self.agents = agents
        self.guard = guard
        self.fault_rate = fault_rate
        self.rng = random.Random(seed)

    def _consult(self, tier: Tier, task: Task, state: NodeState, remaining_s: float) -> NegotiationDecision:
        agent = self.agents[tier]
        dec = agent.decide(task, state).model_copy()
        dec.decided_by = tier

        # Stress test: replace the agent's decision with a random allowed action
        if self.fault_rate > 0 and self.rng.random() < self.fault_rate:
            act = self.rng.choice(sorted(ALLOWED[tier], key=lambda a: a.value))
            dec = NegotiationDecision(
                action=act, target_tier=_TARGET.get(act, tier if act == Action.EXECUTE_LOCAL else None),
                reason="[FAULT-INJECTED] simulated unreliable SLM output", fault_injected=True,
                decided_by=tier, decision_latency_s=dec.decision_latency_s,
            )

        # Format guard: action must be legal for this tier
        if dec.action not in ALLOWED[tier]:
            lat, inj = dec.decision_latency_s, dec.fault_injected
            dec = agent._rule_based_fallback(task, state, f"illegal action {dec.action.value} for {tier.value}")
            dec.decided_by, dec.decision_latency_s, dec.fault_injected = tier, lat, inj

        # Resource-feasibility guard (the novelty)
        if self.guard is not None:
            dec = self.guard.check(tier, task, state, dec, remaining_s)
        return dec

    def negotiate(self, task: Task, states: Dict[Tier, NodeState], sim_time: float = 0.0
                  ) -> Tuple[Optional[Tier], float, List[NegotiationDecision]]:
        history: List[NegotiationDecision] = []
        comm = 0.0

        # Round 1: Mist
        dec = self._consult(Tier.MIST, task, states[Tier.MIST], task.deadline_s)
        history.append(dec)
        if dec.action == Action.EXECUTE_LOCAL:
            return Tier.MIST, comm, history

        comm += NetworkChannel.get_transfer_time(Tier.MIST, Tier.FOG, task.data_size_mb)
        if task.deadline_s - comm <= 0:
            history.append(NegotiationDecision(action=Action.REJECT, is_fallback=True,
                           reason=f"Transfer delay ({comm:.2f}s) exceeded deadline ({task.deadline_s:.2f}s)"))
            return None, comm, history

        # Round 2: Fog
        dec = self._consult(Tier.FOG, task, states[Tier.FOG], task.deadline_s - comm)
        history.append(dec)
        if dec.action == Action.EXECUTE_LOCAL:
            return Tier.FOG, comm, history
        if dec.action == Action.REJECT:
            return None, comm, history

        comm += NetworkChannel.get_transfer_time(Tier.FOG, Tier.CLOUD, task.data_size_mb)
        if task.deadline_s - comm <= 0:
            history.append(NegotiationDecision(action=Action.REJECT, is_fallback=True,
                           reason=f"Cumulative transfer delay ({comm:.2f}s) exceeded deadline ({task.deadline_s:.2f}s)"))
            return None, comm, history

        # Round 3: Cloud
        dec = self._consult(Tier.CLOUD, task, states[Tier.CLOUD], task.deadline_s - comm)
        history.append(dec)
        if dec.action == Action.EXECUTE_LOCAL:
            return Tier.CLOUD, comm, history
        return None, comm, history