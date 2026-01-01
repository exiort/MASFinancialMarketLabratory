from __future__ import annotations
from typing import List
import random

from .agent import Agent
from agents.models import AgentView, AgentConstants, AgentFeedback
from agents.intents import AgentIntent, PlaceOrderIntent

from environment.views import AccountView
from environment.models.order import OrderType, Side


class NoiseTrader(Agent):
    rng:random.Random
    p_trade:float
    p_buy:float
    p_market_order:float
    min_quantity:int
    max_quantity:int
    price_offset_ticks:int
    
    __intent_id:int

    
    def __init__(
            self,
            agent_id:int,
            account_view:AccountView,
            constants:AgentConstants,
            rng:random.Random,
            p_trade:float,
            p_buy:float,
            p_market_order:float,
            min_quantity:int,
            max_quantity:int,
            price_offset_ticks:int
    ):
        super().__init__(agent_id, account_view, constants)

        self.rng = rng
        self.p_trade = p_trade
        self.p_buy = p_buy
        self.p_market_order = p_market_order
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.price_offset_ticks = price_offset_ticks

        self.__intent_id = 0


    @property
    def intent_id(self) -> int:
        intent_id = self.__intent_id
        self.__intent_id += 1

        return intent_id
        
    
    def decide(self, view:AgentView) -> List[AgentIntent]:
        assert view.agent_id == self.agent_id
        assert view.market_data_view is not None
        assert view.economy_insight_view is None
        
        cash = self.account_view.cash
        shares = self.account_view.shares
        
        if self.rng.random() > self.p_trade:
            return []

        side = Side.BUY if self.rng.random() < self.p_buy else Side.SELL
        quantity = self.rng.randint(self.min_quantity, self.max_quantity)

        if side == Side.SELL:
            if shares < quantity:
                return []

        
        order_type = OrderType.MARKET if self.rng.random() < self.p_market_order else OrderType.LIMIT 

        price = None
        
        if order_type == OrderType.MARKET:
            mid_price = view.market_data_view.mid_price
            if mid_price is not None:
                estimated_required_cash = mid_price * quantity * (1 + self.constants.fee_rate) 
                if cash < estimated_required_cash:
                    return []

        elif order_type == OrderType.LIMIT:
            mid_price = view.market_data_view.mid_price
            if mid_price is None:
                return []

            offset = self.rng.randint(0, self.price_offset_ticks)

            if side == Side.BUY:
                price = max(0.0, mid_price - offset)
            else:
                price = mid_price + offset
            
        else: assert False
                
        return [
            PlaceOrderIntent(
                intent_id = self.intent_id, 
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price
            )
        ]
    

    def update(self, feedback: AgentFeedback) -> None:
        _ = feedback
