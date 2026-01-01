from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

from environment.views import OrderView, DepositView



@dataclass(frozen=True)
class AgentFeedback:
    agent_id:int

    order_results:Dict[int, Optional[OrderView]] #intent_id -> OrderView
    deposit_results:Dict[int, Optional[DepositView]] #intent_id -> DepositView
