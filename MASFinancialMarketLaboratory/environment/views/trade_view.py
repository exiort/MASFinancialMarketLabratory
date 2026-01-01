from __future__ import annotations
from dataclasses import dataclass



@dataclass(frozen=True)
class TradeView:
    trade_id:int

    timestamp:float
    macro_tick:int
    micro_tick:int

    price:float
    quantity:int

    fee:float
