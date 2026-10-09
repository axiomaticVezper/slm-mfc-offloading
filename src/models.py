from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Tier(str, Enum):
    MIST = "mist"
    FOG = "fog"
    CLOUD = "cloud"


class Action(str, Enum):
    EXECUTE_LOCAL = "execute_local"
    OFFLOAD_TO_FOG = "offload_to_fog"
    OFFLOAD_TO_CLOUD = "offload_to_cloud"
    REJECT = "reject"


class Task(BaseModel):
    task_id: str
    cpu_mflops: float = Field(..., description="Compute requirement in Mega-FLOPS")
    memory_mb: float = Field(..., description="Memory requirement in MB")
    data_size_mb: float = Field(..., description="Data transfer size in MB")
    deadline_s: float = Field(..., description="Strict latency deadline in seconds")
    arrival_time: float = Field(default=0.0, description="Task generation timestamp")


class NodeState(BaseModel):
    tier: Tier
    cpu_free_pct: float = Field(..., ge=0, le=100)
    mem_free_mb: float = Field(..., ge=0)
    bandwidth_mbps: float = Field(..., ge=0)
    queue_length: int = Field(default=0, ge=0)


class NegotiationDecision(BaseModel):
    action: Action
    target_tier: Optional[Tier] = None
    reason: str = "Standard decision"
    is_fallback: bool = False          # schema/exception fallback (format guard)
    guard_overridden: bool = False     # resource-feasibility guard changed the decision
    fault_injected: bool = False       # decision replaced by fault-injection stress test
    decided_by: Optional[Tier] = None  # which tier's agent produced this decision
    decision_latency_s: float = 0.0


class TaskResult(BaseModel):
    task_id: str
    policy: str
    executed_tier: Optional[Tier]
    arrival_time: float
    start_time: float
    finish_time: float
    transfer_delay: float
    queue_delay: float
    compute_duration: float
    total_turnaround: float
    deadline: float
    met_deadline: bool
    energy_joules: float
    fallback_used: bool = False
    guard_override: bool = False
    mem_violation: bool = False        # task placed on a node without enough free memory
    decision_latency_s: float = 0.0    # measured SLM inference time (informational)
    inference_charge_s: float = 0.0    # inference time actually added to simulated time