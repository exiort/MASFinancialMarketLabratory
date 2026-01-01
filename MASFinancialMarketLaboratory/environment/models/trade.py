from __future__ import annotations
from dataclasses import dataclass

from environment.views import TradeView
from environment.configs import get_environment_configuration



@dataclass(frozen=True)
class Trade:
    trade_id:int
    
    timestamp:float
    macro_tick:int
    micro_tick:int

    seller_agent_id:int
    sell_order_id:int

    buyer_agent_id:int
    buy_order_id:int

    price:int
    quantity:int
    
    fee:int

    
    def create_view(self) -> TradeView:
        ENV_CONFIG = get_environment_configuration()
        
        return TradeView(
            trade_id=self.trade_id,
            timestamp=self.timestamp,
            macro_tick=self.macro_tick,
            micro_tick=self.micro_tick,
            price=self.price / ENV_CONFIG.PRICE_SCALE,
            quantity=self.quantity,
            fee=self.fee / ENV_CONFIG.PRICE_SCALE
        )
