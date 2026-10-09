from dataclasses import dataclass, field
from typing import Dict, List, Optional

import simpy

from models import Tier, Task, NodeState, TaskResult
from network import NetworkChannel
from llm_agent import SLMAgent
from negotiation import NegotiationProtocol
from guard import ResourceGuard


class ComputeNode:
    """A computational tier (Mist, Fog or Cloud)."""

    def __init__(self, env, tier, capacity_mflops_s, mem_mb, cores, active_w, idle_w):
        self.env = env
        self.tier = tier
        self.capacity_mflops_s = capacity_mflops_s
        self.total_mem_mb = mem_mb
        self.used_mem_mb = 0.0
        self.resource = simpy.Resource(env, capacity=cores)
        self.active_w = active_w
        self.idle_w = idle_w      # defined but not used in the per-task energy model
        self.busy_time = 0.0

    def get_state(self, bw_mbps: float) -> NodeState:
        cores = self.resource.capacity
        in_use = self.resource.count
        return NodeState(
            tier=self.tier,
            cpu_free_pct=max(0.0, 100.0 * (1.0 - in_use / cores)),
            mem_free_mb=max(0.0, self.total_mem_mb - self.used_mem_mb),
            bandwidth_mbps=bw_mbps,
            queue_length=len(self.resource.queue) + in_use,
        )


@dataclass
class Placement:
    tier: Optional[Tier]
    transfer_delay: float = 0.0
    path: List[Tier] = field(default_factory=lambda: [Tier.MIST])
    fallback_used: bool = False
    guard_override: bool = False
    decision_latency_s: float = 0.0
    inference_charge_s: float = 0.0


class MFCSimulator:
    """Discrete-event Mist-Fog-Cloud simulator.

    policies: local_only | greedy | static_threshold | always_fog | slm_negotiation
    use_guard / fault_rate / inference_cost only affect slm_negotiation.
    inference_cost: {Tier: seconds} charged to simulated time for each agent consulted.
    """

    def __init__(self, policy="slm_negotiation", agents=None, use_guard=False,
                 fault_rate=0.0, seed=0, inference_cost=None):
        self.env = simpy.Environment()
        self.policy = policy
        self.results: List[TaskResult] = []
        self.inference_cost = inference_cost or {}

        self.nodes: Dict[Tier, ComputeNode] = {
            Tier.MIST: ComputeNode(self.env, Tier.MIST, 1200.0, 512.0, 1, 1.2, 0.1),
            Tier.FOG: ComputeNode(self.env, Tier.FOG, 7500.0, 4096.0, 2, 20.0, 4.0),
            Tier.CLOUD: ComputeNode(self.env, Tier.CLOUD, 35000.0, 32768.0, 8, 180.0, 25.0),
        }
        self.agents = agents or {t: SLMAgent(t) for t in Tier}
        self.protocol = NegotiationProtocol(
            self.agents, guard=ResourceGuard() if use_guard else None,
            fault_rate=fault_rate, seed=seed,
        )

    def _get_states(self):
        return {
            Tier.MIST: self.nodes[Tier.MIST].get_state(30.0),
            Tier.FOG: self.nodes[Tier.FOG].get_state(250.0),
            Tier.CLOUD: self.nodes[Tier.CLOUD].get_state(500.0),
        }

    def _decide(self, task: Task) -> Placement:
        states = self._get_states()
        T = NetworkChannel.get_transfer_time

        if self.policy == "local_only":
            return Placement(Tier.MIST)

        if self.policy == "always_fog":
            return Placement(Tier.FOG, T(Tier.MIST, Tier.FOG, task.data_size_mb), [Tier.MIST, Tier.FOG])

        if self.policy == "greedy":
            if task.cpu_mflops > 3000:
                return Placement(Tier.CLOUD, T(Tier.MIST, Tier.CLOUD, task.data_size_mb), [Tier.MIST, Tier.CLOUD])
            if task.cpu_mflops > 1000:
                return Placement(Tier.FOG, T(Tier.MIST, Tier.FOG, task.data_size_mb), [Tier.MIST, Tier.FOG])
            return Placement(Tier.MIST)

        if self.policy == "static_threshold":
            if task.cpu_mflops > 800 or states[Tier.MIST].cpu_free_pct < 35:
                if states[Tier.FOG].cpu_free_pct > 30 and task.cpu_mflops < 3500:
                    return Placement(Tier.FOG, T(Tier.MIST, Tier.FOG, task.data_size_mb), [Tier.MIST, Tier.FOG])
                return Placement(Tier.CLOUD, T(Tier.MIST, Tier.CLOUD, task.data_size_mb), [Tier.MIST, Tier.CLOUD])
            return Placement(Tier.MIST)

        # slm_negotiation
        tier, comm, history = self.protocol.negotiate(task, states, self.env.now)
        consulted = [d.decided_by for d in history if d.decided_by is not None]
        path = {Tier.MIST: [Tier.MIST], Tier.FOG: [Tier.MIST, Tier.FOG],
                Tier.CLOUD: [Tier.MIST, Tier.FOG, Tier.CLOUD]}.get(tier, [Tier.MIST])
        return Placement(
            tier=tier, transfer_delay=comm, path=path,
            fallback_used=any(d.is_fallback for d in history),
            guard_override=any(d.guard_overridden for d in history),
            decision_latency_s=sum(d.decision_latency_s for d in history),
            inference_charge_s=sum(self.inference_cost.get(t, 0.0) for t in consulted),
        )

    def handle_task(self, task: Task):
        arrival_t = self.env.now
        p = self._decide(task)

        if p.tier is None:   # rejected during negotiation
            self.results.append(TaskResult(
                task_id=task.task_id, policy=self.policy, executed_tier=None,
                arrival_time=arrival_t, start_time=arrival_t,
                finish_time=arrival_t + p.inference_charge_s + p.transfer_delay,
                transfer_delay=p.transfer_delay, queue_delay=0.0, compute_duration=0.0,
                total_turnaround=p.inference_charge_s + p.transfer_delay,
                deadline=task.deadline_s, met_deadline=False, energy_joules=0.0,
                fallback_used=p.fallback_used, guard_override=p.guard_override,
                decision_latency_s=p.decision_latency_s, inference_charge_s=p.inference_charge_s,
            ))
            return

        if p.inference_charge_s > 0:
            yield self.env.timeout(p.inference_charge_s)
        if p.transfer_delay > 0:
            yield self.env.timeout(p.transfer_delay)

        node = self.nodes[p.tier]
        queue_start = self.env.now
        with node.resource.request() as req:
            yield req
            exec_start = self.env.now
            queue_delay = exec_start - queue_start
            compute_duration = task.cpu_mflops / node.capacity_mflops_s
            mem_violation = (node.used_mem_mb + task.memory_mb) > node.total_mem_mb
            node.used_mem_mb += task.memory_mb
            yield self.env.timeout(compute_duration)
            node.used_mem_mb = max(0.0, node.used_mem_mb - task.memory_mb)
            node.busy_time += compute_duration
            finish_t = self.env.now

        turnaround = finish_t - arrival_t
        # Transmission energy summed over every hop actually travelled
        tx_energy = sum(NetworkChannel.get_transfer_energy(p.path[i], p.path[i + 1], task.data_size_mb)
                        for i in range(len(p.path) - 1))
        energy = tx_energy + compute_duration * node.active_w

        self.results.append(TaskResult(
            task_id=task.task_id, policy=self.policy, executed_tier=p.tier,
            arrival_time=arrival_t, start_time=exec_start, finish_time=finish_t,
            transfer_delay=p.transfer_delay, queue_delay=queue_delay,
            compute_duration=compute_duration, total_turnaround=turnaround,
            deadline=task.deadline_s, met_deadline=turnaround <= task.deadline_s,
            energy_joules=energy, fallback_used=p.fallback_used, guard_override=p.guard_override,
            mem_violation=mem_violation, decision_latency_s=p.decision_latency_s,
            inference_charge_s=p.inference_charge_s,
        ))

    def run_workload(self, tasks: List[Task]):
        for task in tasks:
            if task.arrival_time > self.env.now:
                yield self.env.timeout(task.arrival_time - self.env.now)
            self.env.process(self.handle_task(task))

    def run(self, tasks: List[Task]) -> List[TaskResult]:
        self.env.process(self.run_workload(tasks))
        self.env.run()
        return self.results