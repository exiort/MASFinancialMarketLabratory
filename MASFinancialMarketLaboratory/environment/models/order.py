from __future__ import annotations
from typing import Optional, Dict
from enum import Enum, auto
from dataclasses import dataclass, field

from environment.views import OrderView

from .trade import Trade



class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()


class Side(Enum):
    BUY = auto()
    SELL = auto()


class OrderLifecycle(Enum):
    NEW = auto()
    WORKING = auto()
    DONE = auto()


class OrderEndReasons(Enum):
    NONE = auto()
    FILLED = auto()
    CANCELLED = auto()
    EXPIRED = auto()
    REJECTED_INSUFFICIENT_FUND = auto()
    REJECTED_INSUFFICIENT_MARKET_DEPTH = auto()
    KILLED_WASH_TRADE = auto()
    
    
@dataclass
class Order:
    order_id:int
    agent_id:int

    timestamp:float
    macro_tick:int
    micro_tick:int

    order_type:OrderType
    side:Side

    quantity:int
    price:Optional[int]

    lifecycle:OrderLifecycle
    end_reason:OrderEndReasons
    
    remaining_quantity:int = field(init=False)
    
    trades:Dict[int, Trade] = field(default_factory=dict)

    
    def __post_init__(self) -> None:
        self.remaining_quantity = self.quantity


    def create_view(self) -> OrderView:
        return OrderView(self)
