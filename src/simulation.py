import random
from typing import Dict, List, Optional, Tuple
import simpy
from models import Tier, Task, NodeState, TaskResult
from network import NetworkChannel
from llm_agent import SLMAgent
from negotiation import NegotiationProtocol

class ComputeNode:
    """Represents a computational tier (Mist, Fog, or Cloud)."""
    def __init__(self, env: simpy.Environment, tier: Tier, capacity_mflops_s: float, mem_mb: float, cores: int, active_w: float, idle_w: float):
        self.env = env
        self.tier = tier
        self.capacity_mflops_s = capacity_mflops_s
        self.total_mem_mb = mem_mb
        self.used_mem_mb = 0.0
        self.resource = simpy.Resource(env, capacity=cores)
        self.active_w = active_w
        self.idle_w = idle_w
        self.busy_time = 0.0

    def get_state(self, bw_mbps: float) -> NodeState:
        cores = self.resource.capacity
        in_use = self.resource.count
        cpu_free = max(0.0, 100.0 * (1.0 - (in_use / cores)))
        mem_free = max(0.0, self.total_mem_mb - self.used_mem_mb)
        queue_len = len(self.resource.queue) + in_use
        return NodeState(
            tier=self.tier,
            cpu_free_pct=cpu_free,
            mem_free_mb=mem_free,
            bandwidth_mbps=bw_mbps,
            queue_length=queue_len
        )

class MFCSimulator:
    """Discrete-Event Simulator for Mist-Fog-Cloud Adaptive Offloading."""

    def __init__(self, policy: str = "slm_negotiation", agents: Optional[Dict[Tier, SLMAgent]] = None):
        self.env = simpy.Environment()
        self.policy = policy
        self.results: List[TaskResult] = []

        # Tier node configurations
        self.nodes: Dict[Tier, ComputeNode] = {
            Tier.MIST: ComputeNode(self.env, Tier.MIST, capacity_mflops_s=1200.0, mem_mb=512.0, cores=1, active_w=1.2, idle_w=0.1),
            Tier.FOG: ComputeNode(self.env, Tier.FOG, capacity_mflops_s=7500.0, mem_mb=4096.0, cores=2, active_w=20.0, idle_w=4.0),
            Tier.CLOUD: ComputeNode(self.env, Tier.CLOUD, capacity_mflops_s=35000.0, mem_mb=32768.0, cores=8, active_w=180.0, idle_w=25.0),
        }

        self.agents = agents or {
            Tier.MIST: SLMAgent(Tier.MIST),
            Tier.FOG: SLMAgent(Tier.FOG),
            Tier.CLOUD: SLMAgent(Tier.CLOUD)
        }
        self.protocol = NegotiationProtocol(self.agents)

    def _get_states(self) -> Dict[Tier, NodeState]:
        return {
            Tier.MIST: self.nodes[Tier.MIST].get_state(bw_mbps=30.0),
            Tier.FOG: self.nodes[Tier.FOG].get_state(bw_mbps=250.0),
            Tier.CLOUD: self.nodes[Tier.CLOUD].get_state(bw_mbps=500.0),
        }

    def _decide_tier(self, task: Task) -> Tuple[Optional[Tier], float, bool]:
        states = self._get_states()

        if self.policy == "local_only":
            return Tier.MIST, 0.0, False

        elif self.policy == "greedy":
            # Always offload to tier with most available compute
            if task.cpu_mflops > 3000:
                t_delay = NetworkChannel.get_transfer_time(Tier.MIST, Tier.CLOUD, task.data_size_mb)
                return Tier.CLOUD, t_delay, False
            elif task.cpu_mflops > 1000:
                t_delay = NetworkChannel.get_transfer_time(Tier.MIST, Tier.FOG, task.data_size_mb)
                return Tier.FOG, t_delay, False
            return Tier.MIST, 0.0, False

        elif self.policy == "static_threshold":
            # Threshold rule: Mist offloads if CPU > 65% or task > 800 MFLOPS
            mist_state = states[Tier.MIST]
            if task.cpu_mflops > 800 or mist_state.cpu_free_pct < 35:
                fog_state = states[Tier.FOG]
                if fog_state.cpu_free_pct > 30 and task.cpu_mflops < 3500:
                    t_delay = NetworkChannel.get_transfer_time(Tier.MIST, Tier.FOG, task.data_size_mb)
                    return Tier.FOG, t_delay, False
                else:
                    t_delay = NetworkChannel.get_transfer_time(Tier.MIST, Tier.CLOUD, task.data_size_mb)
                    return Tier.CLOUD, t_delay, False
            return Tier.MIST, 0.0, False

        elif self.policy == "slm_negotiation":
            chosen_tier, t_delay, history = self.protocol.negotiate(task, states, self.env.now)
            used_fallback = any(h.is_fallback for h in history)
            return chosen_tier, t_delay, used_fallback

        return Tier.MIST, 0.0, False

    def handle_task(self, task: Task):
        arrival_t = self.env.now
        chosen_tier, transfer_delay, fallback_used = self._decide_tier(task)

        if chosen_tier is None:
            # Task rejected during negotiation
            self.results.append(TaskResult(
                task_id=task.task_id,
                policy=self.policy,
                executed_tier=None,
                arrival_time=arrival_t,
                start_time=arrival_t,
                finish_time=arrival_t + transfer_delay,
                transfer_delay=transfer_delay,
                queue_delay=0.0,
                compute_duration=0.0,
                total_turnaround=transfer_delay,
                deadline=task.deadline_s,
                met_deadline=False,
                energy_joules=0.0,
                fallback_used=fallback_used
            ))
            return

        # Simulate network transfer delay
        if transfer_delay > 0:
            yield self.env.timeout(transfer_delay)

        target_node = self.nodes[chosen_tier]
        queue_start = self.env.now

        # Request compute resource at chosen tier
        with target_node.resource.request() as req:
            yield req
            exec_start = self.env.now
            queue_delay = exec_start - queue_start

            # Calculate execution duration based on tier capacity
            compute_duration = task.cpu_mflops / target_node.capacity_mflops_s
            target_node.used_mem_mb += task.memory_mb

            yield self.env.timeout(compute_duration)

            target_node.used_mem_mb = max(0.0, target_node.used_mem_mb - task.memory_mb)
            target_node.busy_time += compute_duration
            finish_t = self.env.now

        total_turnaround = finish_t - arrival_t
        met_deadline = total_turnaround <= task.deadline_s

        # Energy calculation
        transfer_energy = NetworkChannel.get_transfer_energy(Tier.MIST, chosen_tier, task.data_size_mb) if chosen_tier != Tier.MIST else 0.0
        compute_energy = compute_duration * target_node.active_w
        total_energy = transfer_energy + compute_energy

        self.results.append(TaskResult(
            task_id=task.task_id,
            policy=self.policy,
            executed_tier=chosen_tier,
            arrival_time=arrival_t,
            start_time=exec_start,
            finish_time=finish_t,
            transfer_delay=transfer_delay,
            queue_delay=queue_delay,
            compute_duration=compute_duration,
            total_turnaround=total_turnaround,
            deadline=task.deadline_s,
            met_deadline=met_deadline,
            energy_joules=total_energy,
            fallback_used=fallback_used
        ))

    def run_workload(self, tasks: List[Task]):
        for task in tasks:
            # Inter-arrival delay
            if task.arrival_time > self.env.now:
                yield self.env.timeout(task.arrival_time - self.env.now)
            self.env.process(self.handle_task(task))
