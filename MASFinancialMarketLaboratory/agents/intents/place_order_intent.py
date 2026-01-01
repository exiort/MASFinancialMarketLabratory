from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .agent_intent import AgentIntent

from environment.models.order import OrderType, Side 



@dataclass(frozen=True)
class PlaceOrderIntent(AgentIntent):
    side:Side
    order_type:OrderType
    quantity:int
    price:Optional[float]
    
