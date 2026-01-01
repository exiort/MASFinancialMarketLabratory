from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from environment.views import MarketDataView, EconomyInsightView



@dataclass(frozen=True)
class AgentView:
    agent_id:int

    timestamp:float
    macro_tick:int
    micro_tick:int

    market_data_view:Optional[MarketDataView]
    economy_insight_view:Optional[EconomyInsightView]
    
