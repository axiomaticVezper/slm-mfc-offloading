from typing import Dict, List, Tuple, Optional
from models import Tier, Action, Task, NodeState, NegotiationDecision
from network import NetworkChannel
from llm_agent import SLMAgent

class NegotiationProtocol:
    """Manages multi-round negotiation across Mist, Fog, and Cloud agents."""

    def __init__(self, agents: Dict[Tier, SLMAgent]):
        self.agents = agents

    def negotiate(
        self,
        task: Task,
        states: Dict[Tier, NodeState],
        sim_time: float = 0.0
    ) -> Tuple[Optional[Tier], float, List[NegotiationDecision]]:
        """
        Conducts hierarchical negotiation:
        Round 1: Mist agent assesses local execution vs offload request.
        Round 2: Fog agent counters or forwards.
        Round 3: Cloud agent allocates.
        """
        history: List[NegotiationDecision] = []
        comm_overhead_s = 0.0

        # --- ROUND 1: MIST ---
        mist_agent = self.agents[Tier.MIST]
        mist_dec = mist_agent.decide(task, states[Tier.MIST])
        history.append(mist_dec)

        if mist_dec.action == Action.EXECUTE_LOCAL:
            return Tier.MIST, comm_overhead_s, history

        # Mist requested offload to Fog
        comm_overhead_s += NetworkChannel.get_transfer_time(Tier.MIST, Tier.FOG, task.data_size_mb)
        
        # Check deadline viability
        remaining_deadline = task.deadline_s - comm_overhead_s
        if remaining_deadline <= 0:
            reject_dec = NegotiationDecision(
                action=Action.REJECT,
                reason=f"Transfer delay ({comm_overhead_s:.2f}s) exceeded deadline ({task.deadline_s:.2f}s)",
                is_fallback=True
            )
            history.append(reject_dec)
            return None, comm_overhead_s, history

        # --- ROUND 2: FOG ---
        fog_agent = self.agents[Tier.FOG]
        fog_dec = fog_agent.decide(task, states[Tier.FOG])
        history.append(fog_dec)

        if fog_dec.action == Action.EXECUTE_LOCAL:
            return Tier.FOG, comm_overhead_s, history

        if fog_dec.action == Action.REJECT:
            return None, comm_overhead_s, history

        # Fog requested forward to Cloud
        comm_overhead_s += NetworkChannel.get_transfer_time(Tier.FOG, Tier.CLOUD, task.data_size_mb)
        remaining_deadline = task.deadline_s - comm_overhead_s
        if remaining_deadline <= 0:
            reject_dec = NegotiationDecision(
                action=Action.REJECT,
                reason=f"Cumulative transfer delay ({comm_overhead_s:.2f}s) exceeded deadline ({task.deadline_s:.2f}s)",
                is_fallback=True
            )
            history.append(reject_dec)
            return None, comm_overhead_s, history

        # --- ROUND 3: CLOUD ---
        cloud_agent = self.agents[Tier.CLOUD]
        cloud_dec = cloud_agent.decide(task, states[Tier.CLOUD])
        history.append(cloud_dec)

        if cloud_dec.action == Action.EXECUTE_LOCAL:
            return Tier.CLOUD, comm_overhead_s, history

        # Cloud rejected
        return None, comm_overhead_s, history
