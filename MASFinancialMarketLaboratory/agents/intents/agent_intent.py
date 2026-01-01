from __future__ import annotations
from abc import ABC
from dataclasses import dataclass



@dataclass(frozen=True)
class AgentIntent(ABC):
    intent_id:int
