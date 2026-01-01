from __future__ import annotations
from dataclasses import dataclass

from .agent_intent import AgentIntent



@dataclass(frozen=True)
class CancelOrderIntent(AgentIntent):
    order_id:int
