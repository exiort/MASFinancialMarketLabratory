from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from environment.configs import get_environment_configuration

from .trade_view import TradeView

if TYPE_CHECKING:
    from environment.models.order import Order, OrderType, Side, OrderLifecycle, OrderEndReasons

    

@dataclass
class OrderView:
    def __init__(self, order:Order) -> None:
        self.__order = order

        
    @property
    def order_id(self) -> int:
        return self.__order.order_id

    
    @property
    def agent_id(self) -> int:
        return self.__order.agent_id

    
    @property
    def timestamp(self) -> float:
        return self.__order.timestamp


    @property
    def macro_tick(self) -> int:
        return self.__order.macro_tick

    
    @property
    def micro_tick(self) -> int:
        return self.__order.micro_tick


    @property
    def order_type(self) -> OrderType:
        return self.__order.order_type


    @property
    def side(self) -> Side:
        return self.__order.side


    @property
    def quantity(self) -> int:
        return self.__order.quantity


    @property
    def price(self) -> Optional[float]:
        if self.__order.price is None:
            return None

        ENV_CONFIG = get_environment_configuration()
        return self.__order.price / ENV_CONFIG.PRICE_SCALE
    

    @property
    def lifecycle(self) -> OrderLifecycle:
        return self.__order.lifecycle


    @property
    def end_reason(self) -> OrderEndReasons:
        return self.__order.end_reason


    @property
    def remaining_quantity(self) -> int:
        return self.__order.remaining_quantity

    
    @property
    def trades(self) -> Tuple[TradeView, ...]:
        trade_views = []
        for trade in self.__order.trades.values():
            trade_views.append(trade.create_view())

        return tuple(trade_views)
