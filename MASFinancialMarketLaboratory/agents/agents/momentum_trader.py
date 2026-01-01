from __future__ import annotations
from typing import List, Dict, Optional

from .agent import Agent
from agents.models import AgentView, AgentConstants, AgentFeedback
from agents.intents import AgentIntent, PlaceOrderIntent

from environment.views import AccountView, OrderView
from environment.models.order import OrderType, Side, OrderLifecycle



class MomentumTrader(Agent):
    #OBSERVE → EVALUATE → ACT
        
    # Signal Parameters
    momentum_window: int
    entry_threshold: float
    exit_threshold: float
    
    # Position Parameters
    aggressiveness: float
    max_exposure_fraction: float
    directional_bias: float
    liquidity_baseline: int
    
    price_history: List[float]
    active_orders: Dict[int, OrderView]
    
    __intent_id: int
    
    
    def __init__(
        self,
        agent_id: int,
        account_view: AccountView,
        constants: AgentConstants,
        # Signal parameters
        momentum_window: int,
        entry_threshold: float,
        exit_threshold: float,
        # Position parameters
        aggressiveness: float,
        max_exposure_fraction: float,
        directional_bias: float,
        liquidity_baseline: int
    ) -> None:
        super().__init__(agent_id, account_view, constants)
        
        self.momentum_window = momentum_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.aggressiveness = aggressiveness
        self.max_exposure_fraction = max_exposure_fraction
        self.directional_bias = directional_bias
        self.liquidity_baseline = liquidity_baseline
        
        self.price_history = []
        self.active_orders = {}
        
        self.__intent_id = 0
    
    
    @property
    def intent_id(self) -> int:
        intent_id = self.__intent_id
        self.__intent_id += 1
        return intent_id

    
    def decide(self, view: AgentView) -> List[AgentIntent]:
        assert view.agent_id == self.agent_id
        assert view.market_data_view is not None
        
        # OBSERVE: Update price history
        self._update_price_history(view)
        
        # EVALUATE: Calculate momentum and evaluate signal
        momentum = self._calculate_momentum_signal()
        biased_momentum = self._apply_directional_bias(momentum)
        signal = self._evaluate_signal(biased_momentum)
        
        # ACT: Execute based on signal
        return self._act_on_signal(signal, view)
    
    
    def update(self, feedback: AgentFeedback) -> None:
        for _, order_view in feedback.order_results.items():
            if order_view is not None:
                self.active_orders[order_view.order_id] = order_view
        
        self._cleanup_done_orders()
    
    
    def _update_price_history(self, view: AgentView) -> None:
        current_price = self._get_current_price(view)
        if current_price is None:
            return
        
        self.price_history.append(current_price)
        
        if len(self.price_history) > self.momentum_window:
            self.price_history.pop(0)
    
    
    def _get_current_price(self, view: AgentView) -> Optional[float]:
        md = view.market_data_view
        assert md is not None
        
        if md.last_traded_price is not None:
            return md.last_traded_price
        
        if md.mid_price is not None:
            return md.mid_price
        
        if md.micro_price is not None:
            return md.micro_price
        
        return None

    
    def _calculate_momentum_signal(self) -> float:
        #Weighted momentum - recent prices matter more.

        if len(self.price_history) < 2:
            return 0.0
        
        total_weight = 0.0
        weighted_return = 0.0
        
        for i in range(1, len(self.price_history)):
            price_prev = self.price_history[i-1]
            price_curr = self.price_history[i]
            
            if price_prev > 0:
                ret = (price_curr - price_prev) / price_prev
                weight = i
                
                weighted_return += ret * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_return / total_weight
        
        return 0.0
    
    
    def _apply_directional_bias(self, momentum: float) -> float:
        biased_momentum = momentum + self.directional_bias
        return biased_momentum
    
    
    def _evaluate_signal(self, biased_momentum: float) -> str:
        if biased_momentum > self.entry_threshold:
            return 'ENTER_LONG'
        
        if biased_momentum < -self.entry_threshold:
            return 'ENTER_SHORT'
        
        if abs(biased_momentum) < self.exit_threshold:
            return 'EXIT'
        
        return 'HOLD'

    
    def _calculate_liquidity_amplifier(self, view: AgentView) -> float:
        #Amplify aggressiveness when market liquidity is low.
        #Low liquidity → higher amplifier → larger positions
        
        md = view.market_data_view
        assert md is not None

        total_liquidity = 0
        
        if md.L1_bids is not None:
            # L1_bids = (price, size, #orders)
            total_liquidity += md.L1_bids[1]
        
        if md.L1_asks is not None:
            # L1_asks = (price, size, #orders)
            total_liquidity += md.L1_asks[1]
        
        if total_liquidity == 0:
            return 1.0
        
        liquidity_ratio = total_liquidity / self.liquidity_baseline
        
        # Clamped to [0.5, 2.0]
        amplifier = 1.0 / max(0.5, min(2.0, liquidity_ratio))
        
        return amplifier
    
        
    def _act_on_signal(self, signal: str, view: AgentView) -> List[AgentIntent]:
        if signal == 'ENTER_LONG':
            return self._enter_long(view)
        
        elif signal == 'ENTER_SHORT':
            return self._enter_short(view)
        
        elif signal == 'EXIT':
            return self._exit_position(view)
        
        else:
            return []
    
    
    def _enter_long(self, view: AgentView) -> List[AgentIntent]:
        cash = self.account_view.cash
        shares = self.account_view.shares
        
        # Calculate position size
        current_price = self._get_current_price(view)
        if current_price is None:
            return []
        
        # Total wealth
        total_wealth = cash + shares * current_price
        
        # Calculate liquidity amplifier
        liquidity_amp = self._calculate_liquidity_amplifier(view)
        
        # Calculate target exposure
        effective_aggressiveness = self.aggressiveness * liquidity_amp
        target_value = total_wealth * self.max_exposure_fraction * effective_aggressiveness
        target_shares = int(target_value / current_price)
        
        # How much more to buy
        additional_shares = target_shares - shares
        
        if additional_shares <= 0:
            return []  
        
        cost = additional_shares * current_price * (1 + self.constants.fee_rate)
        if cost > cash:
            additional_shares = int(cash / (current_price * (1 + self.constants.fee_rate)))
        
        if additional_shares <= 0:
            return []
        
        return [
            PlaceOrderIntent(
                intent_id=self.intent_id,
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=additional_shares,
                price=None
            )
        ]
    
    
    def _enter_short(self, view: AgentView) -> List[AgentIntent]:
        #'Short' by selling shares (reduce long position).
        
        shares = self.account_view.shares
        
        if shares <= 0:
            return []
        
        # Calculate how much to sell
        current_price = self._get_current_price(view)
        if current_price is None:
            return []
        
        # Liquidity amplifier
        liquidity_amp = self._calculate_liquidity_amplifier(view)
        
        # Calculate target: reduce position aggressively
        effective_aggressiveness = self.aggressiveness * liquidity_amp
        
        sell_quantity = int(shares * effective_aggressiveness)
        sell_quantity = min(sell_quantity, shares)  # Can't sell more than we have
        
        if sell_quantity <= 0:
            return []
        
        return [
            PlaceOrderIntent(
                intent_id=self.intent_id,
                side=Side.SELL,
                order_type=OrderType.MARKET,
                quantity=sell_quantity,
                price=None
            )
        ]
    
    
    def _exit_position(self, view: AgentView) -> List[AgentIntent]:
        _ = view
        shares = self.account_view.shares
            
        # Aggressive liquidation
        if shares > 0:
            return [
                PlaceOrderIntent(
                    intent_id=self.intent_id,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=shares,
                    price=None
                )
            ]
        
        return []
    
        
    def _cleanup_done_orders(self) -> None:
        for order_id, order_view in list(self.active_orders.items()):
            if order_view.lifecycle == OrderLifecycle.DONE:
                del self.active_orders[order_id]
