from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from enum import Enum

from .agent import Agent
from agents.models import AgentView, AgentConstants, AgentFeedback
from agents.intents import AgentIntent, PlaceOrderIntent, CreateDepositIntent

from environment.views import AccountView, OrderView, DepositView
from environment.models.order import OrderType, Side, OrderLifecycle


class Intention(Enum):
    WAIT = "WAIT"
    ACCUMULATE = "ACCUMULATE"
    DISTRIBUTE = "DISTRIBUTE"
    PARK_CAPITAL = "PARK_CAPITAL"


class ValueInvestor(Agent):    
    #Belief Parameters
    initial_optimism: float
    stubbornness: float
    belief_update_rate: float
    
    #Decision Parameters
    mispricing_threshold: float
    max_position_fraction: float
    deposit_affinity: float
    deposit_allocation_fraction: float
    deposit_horizon_preference: float
    
    #Trading Parameters
    max_order_size: int
    patience_discount: float
    patience_premium: float
    
    #Belief State
    optimism: float
    belief: float
    perceived_pnl: float
    initial_capital: float
    current_intention: Intention
    
    #Tracking
    active_deposits: Dict[int, DepositView]
    pending_orders: Dict[int, OrderView]
    last_known_price: Optional[float]
    
    __intent_id: int
    
    
    def __init__(
        self,
        agent_id: int,
        account_view: AccountView,
        constants: AgentConstants,
        initial_optimism: float,
        stubbornness: float,
        belief_update_rate: float,
        mispricing_threshold: float,
        max_position_fraction: float,
        deposit_affinity: float,
        deposit_allocation_fraction: float,
        deposit_horizon_preference: float,
        max_order_size: int,
        patience_discount: float,
        patience_premium: float
    ) -> None:
        super().__init__(agent_id, account_view, constants)
        
        # Store parameters
        self.initial_optimism = initial_optimism
        self.stubbornness = stubbornness
        self.belief_update_rate = belief_update_rate
        self.mispricing_threshold = mispricing_threshold
        self.max_position_fraction = max_position_fraction
        self.deposit_affinity = deposit_affinity
        self.deposit_allocation_fraction = deposit_allocation_fraction
        self.deposit_horizon_preference = deposit_horizon_preference
        self.max_order_size = max_order_size
        self.patience_discount = patience_discount
        self.patience_premium = patience_premium
        
        self.optimism = initial_optimism
        self.belief = 100.0  
        self.perceived_pnl = 0.0
        self.initial_capital = account_view.cash + account_view.shares * 100.0
        self.current_intention = Intention.WAIT
        
        self.active_deposits = {}
        self.pending_orders = {}
        self.last_known_price = None
        
        self.__intent_id = 0
    
    
    @property
    def intent_id(self) -> int:
        intent_id = self.__intent_id
        self.__intent_id += 1
        return intent_id
    
    
    def decide(self, view: AgentView) -> List[AgentIntent]:
        #BDI decision loop
        # -Update beliefs
        # -Select intention
        # -Execute intention

        assert view.agent_id == self.agent_id
        assert view.economy_insight_view is not None

        # BDI Loop
        self._update_beliefs(view)
        self._select_intention(view)
        return self._execute_intention(view)
    
    
    def update(self, feedback: AgentFeedback) -> None:
        for _, deposit_view in feedback.deposit_results.items():
            if deposit_view is not None:
                self.active_deposits[deposit_view.deposit_id] = deposit_view
        
        for _, order_view in feedback.order_results.items():
            if order_view is not None:
                self.pending_orders[order_view.order_id] = order_view
        
        
    def _update_beliefs(self, view: AgentView) -> None:
        # -Calculate belief from optimism and TV interval
        # -Update optimism based on performance (with stubbornness friction)
        # -Update perceived PnL
       
        economy = view.economy_insight_view
        assert economy is not None
        
        TV_low, TV_high = economy.tv_interval
        
        self.belief = TV_low + self.optimism * (TV_high - TV_low)
        
        self._update_optimism(view)
        self._update_perceived_pnl(view)
    
    
    def _update_optimism(self, view: AgentView) -> None:
        #Stubbornness resists optimism change.
        _ = view
        
        if self.perceived_pnl > 0:
            # Success → increase optimism
            delta = self.belief_update_rate * (1 - self.stubbornness)
            self.optimism = min(1.0, self.optimism + delta)
        elif self.perceived_pnl < 0:
            # Loss → decrease optimism 
            delta = self.belief_update_rate * (1 - self.stubbornness)
            self.optimism = max(0.0, self.optimism - delta)
    
    
    def _update_perceived_pnl(self, view: AgentView) -> None:
        cash = self.account_view.cash
        shares = self.account_view.shares
        
        market_price = self._estimate_market_price(view)
        if market_price is None:
            market_price = self.belief
        
        portfolio_value = cash + shares * market_price
        
        self.perceived_pnl = portfolio_value - self.initial_capital
    
    
    def _calculate_mispricing(self, view: AgentView) -> float:
        #(belief - market_price) / belief
        #Positive: Market undervalued (BUY signal)
        #Negative: Market overvalued (SELL signal)
       
        market_price = self._estimate_market_price(view)
        if market_price is None or market_price <= 0:
            return 0.0
        
        mispricing = (self.belief - market_price) / self.belief
        return mispricing
    
    
    def _calculate_position_fraction(self) -> float:
        cash = self.account_view.cash
        shares = self.account_view.shares
        
        if self.last_known_price is None:
            return 0.0
        
        total_capital = cash + shares * self.last_known_price
        
        if total_capital <= 0:
            return 0.0
        
        position_value = shares * self.last_known_price
        return position_value / total_capital
    
    
    def _evaluate_deposit_opportunity(self, view: AgentView) -> Optional[Tuple[int, float]]:
        economy = view.economy_insight_view
        assert economy is not None
        deposit_rates = economy.deposit_rates
        
        if not deposit_rates:
            return None
        
        best_term = None
        best_score = -float('inf')
        
        max_term = max(deposit_rates.keys())
        
        for term, rate in deposit_rates.items():
            horizon_penalty = (1 - self.deposit_horizon_preference) * (term / max_term)
            score = rate - horizon_penalty
            
            if score > best_score:
                best_score = score
                best_term = term
                
        if best_term:
            best_rate = deposit_rates[best_term]
        
            if best_rate >= self.deposit_affinity:
                return (best_term, best_rate)
        
        return None

    
    def _select_intention(self, view: AgentView) -> None:
        mispricing = self._calculate_mispricing(view)
        position_fraction = self._calculate_position_fraction()
        deposit_opportunity = self._evaluate_deposit_opportunity(view)
        
        # Prio1: ACCUMULATE 
        if mispricing > self.mispricing_threshold:
            if position_fraction < self.max_position_fraction:
                self.current_intention = Intention.ACCUMULATE
                return
        
        # Prio2: DISTRIBUTE (overvalued + have position)
        if mispricing < -self.mispricing_threshold:
            if position_fraction > 0.1:  # Have something to sell
                self.current_intention = Intention.DISTRIBUTE
                return
        
        # Prio3: PARK_CAPITAL
        if deposit_opportunity is not None:
            cash = self.account_view.cash
            if self.last_known_price is not None:
                total_wealth = cash + self.account_view.shares * self.last_known_price
                if total_wealth > 0 and cash / total_wealth > 0.2:
                    self.current_intention = Intention.PARK_CAPITAL
                    return
        
        self.current_intention = Intention.WAIT
    
    
    def _execute_intention(self, view: AgentView) -> List[AgentIntent]:
        if self.current_intention == Intention.WAIT:
            return self._execute_wait(view)
        elif self.current_intention == Intention.ACCUMULATE:
            return self._execute_accumulate(view)
        elif self.current_intention == Intention.DISTRIBUTE:
            return self._execute_distribute(view)
        elif self.current_intention == Intention.PARK_CAPITAL:
            return self._execute_park_capital(view)
        else:
            return []

        
    def _execute_wait(self, view: AgentView) -> List[AgentIntent]:
        _ = view
        return []
    
    
    def _execute_accumulate(self, view: AgentView) -> List[AgentIntent]:
        cash = self.account_view.cash
        if cash <= 0:
            return []
        
        market_price = self._estimate_market_price(view)
        if market_price is None:
            return []
        
        shares_affordable = int(cash / (market_price * (1 + self.constants.fee_rate)))
        quantity = min(shares_affordable, self.max_order_size)
        
        if quantity <= 0:
            return []
        
        buy_price = market_price * (1 - self.patience_discount)
        
        return [
            PlaceOrderIntent(
                intent_id=self.intent_id,
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=buy_price
            )
        ]
    
    
    def _execute_distribute(self, view: AgentView) -> List[AgentIntent]:
        shares = self.account_view.shares
        if shares <= 0:
            return []
        
        market_price = self._estimate_market_price(view)
        if market_price is None:
            return []
        
        quantity = min(shares, self.max_order_size)
        
        sell_price = market_price * (1 + self.patience_premium)
        
        return [
            PlaceOrderIntent(
                intent_id=self.intent_id,
                side=Side.SELL,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=sell_price
            )
        ]
    
    
    def _execute_park_capital(self, view: AgentView) -> List[AgentIntent]:
        cash = self.account_view.cash
        if cash <= 0:
            return []
        
        deposit_opp = self._evaluate_deposit_opportunity(view)
        if deposit_opp is None:
            return []
        
        term, _ = deposit_opp
        
        deposit_amount = cash * self.deposit_allocation_fraction
        
        if deposit_amount <= 0:
            return []
        
        return [
            CreateDepositIntent(
                intent_id=self.intent_id,
                amount=deposit_amount,
                term=term
            )
        ]
    
        
    def _estimate_market_price(self, view: AgentView) -> Optional[float]:
        md = view.market_data_view
        assert md is not None
        
        if md.mid_price is not None:
            self.last_known_price = md.mid_price
            return md.mid_price
        
        if md.last_traded_price is not None:
            self.last_known_price = md.last_traded_price
            return md.last_traded_price
        
        return self.last_known_price
    
    
    def _cleanup_matured_deposits(self, current_macro_tick: int) -> None:
        for deposit_id, deposit_view in list(self.active_deposits.items()):
            if current_macro_tick >= deposit_view.maturity_macro_tick:
                del self.active_deposits[deposit_id]
    
    
    def _cleanup_done_orders(self) -> None:
        for order_id, order_view in list(self.pending_orders.items()):
            if order_view.lifecycle == OrderLifecycle.DONE:
                del self.pending_orders[order_id]
