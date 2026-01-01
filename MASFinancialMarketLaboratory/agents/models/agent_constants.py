from __future__ import annotations
from dataclasses import dataclass



@dataclass(frozen=True)
class AgentConstants:
    simulation_macro_tick:int
    simulation_micro_tick:int

    fee_rate:float
