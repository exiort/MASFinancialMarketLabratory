from __future__ import annotations
from typing import List, Dict, Optional, Tuple

from .agent import Agent
from agents.models import AgentView, AgentConstants, AgentFeedback
from agents.intents import AgentIntent, PlaceOrderIntent, CancelOrderIntent

from environment.views import AccountView, OrderView
from environment.models.order import OrderType, Side, OrderLifecycle, OrderEndReasons



class MarketMaker(Agent):
    #SURVIVE > STABILIZE > PROVIDE
    
    #Survival Layer
    risk_lower_bound: float
    risk_upper_bound: float
    
    #Stabilize Layer
    target_inventory_fraction: float
    stabilization_tolerance: float
    skew_factor: int
    
    #Provide Layer
    spread_size: int
    order_size: int
    wait_time: int
    
    active_bids: Dict[int, OrderView]
    active_asks: Dict[int, OrderView]
    last_known_price: Optional[float]
    last_quote_tick: Optional[int]
    current_micro_tick: int
    
    __intent_id: int
    
    
    def __init__(
        self,
        agent_id: int,
        account_view: AccountView,
        constants: AgentConstants,
        target_inventory_fraction: float,
        risk_lower_bound: float,
        risk_upper_bound: float,
        stabilization_tolerance: float,
        spread_size: int,
        order_size: int,
        wait_time: int,
        skew_factor: int
    ) -> None:
        super().__init__(agent_id, account_view, constants)
        
        self.target_inventory_fraction = target_inventory_fraction
        self.risk_lower_bound = risk_lower_bound
        self.risk_upper_bound = risk_upper_bound
        self.stabilization_tolerance = stabilization_tolerance
        self.spread_size = spread_size
        self.order_size = order_size
        self.wait_time = wait_time
        self.skew_factor = skew_factor
        
        self.active_bids = {}
        self.active_asks = {}
        self.last_known_price = None
        self.last_quote_tick = None
        self.current_micro_tick = 0
        
        self.last_mid_price: Optional[float] = None
        self.last_inventory_ratio: Optional[float] = None
        self.layer_changed: bool = False
        self.order_filled_this_tick: bool = False
        self._last_layer: str = 'PROVIDE'
        
        self.current_global_tick: int = 0
        self.last_quote_global_tick: Optional[int] = None
        
        self.__intent_id = 0
    
    
    @property
    def intent_id(self) -> int:
        intent_id = self.__intent_id
        self.__intent_id += 1
        return intent_id
    
    
    def decide(self, view: AgentView) -> List[AgentIntent]:
        assert view.agent_id == self.agent_id
        assert view.market_data_view is not None
        assert view.economy_insight_view is not None
        

        self.current_micro_tick = view.micro_tick
        self.current_global_tick = view.macro_tick * self.constants.simulation_micro_tick + view.micro_tick
        
        self.layer_changed = False
        current_layer = self._determine_current_layer()
        
        if self._last_layer != current_layer:
            self.layer_changed = True
        self._last_layer = current_layer
        
        
        new_ratio = self._calculate_inventory_ratio()
        if self.last_inventory_ratio is not None:
            old_direction = self.last_inventory_ratio - self.target_inventory_fraction
            new_direction = new_ratio - self.target_inventory_fraction
        
            if old_direction * new_direction < 0:
                self.layer_changed = True
                
        self.last_inventory_ratio = new_ratio
        
        
        if current_layer == 'SURVIVE':
            return self._survive_layer(view)
        
        if current_layer == 'STABILIZE':
            return self._stabilize_layer(view)
        
        return self._provide_layer(view)
    
    
    def _determine_current_layer(self) -> str:
        if self._should_survive():
            return 'SURVIVE'
        if self._should_stabilize():
            return 'STABILIZE'
        return 'PROVIDE'
    
    
    def update(self, feedback: AgentFeedback) -> None:
        self.order_filled_this_tick = False
        
        for _, order_view in feedback.order_results.items():
            if order_view is None:
                continue
            
            if order_view.side == Side.BUY:
                self.active_bids[order_view.order_id] = order_view
            else:
                self.active_asks[order_view.order_id] = order_view
        
        self._cleanup_done_orders()
    
    
    def _should_survive(self) -> bool:
        inventory_ratio = self._calculate_inventory_ratio()
        
        if inventory_ratio < self.risk_lower_bound:
            return True  # Too many shares
        if inventory_ratio > self.risk_upper_bound:
            return True  # Too much cash
        
        return False
    
    
    def _survive_layer(self, view: AgentView) -> List[AgentIntent]:
        intents = []
        
        intents.extend(self._cancel_all_orders())
        
        inventory_ratio = self._calculate_inventory_ratio()
        
        if inventory_ratio < self.risk_lower_bound:
            # Too many shares - PANIC SELL
            intent = self._create_panic_sell_intent(view)
            if intent:
                intents.append(intent)
        
        elif inventory_ratio > self.risk_upper_bound:
            # Too much cash - PANIC BUY
            intent = self._create_panic_buy_intent(view)
            if intent:
                intents.append(intent)
        
        return intents
    
    
    def _create_panic_sell_intent(self, view: AgentView) -> Optional[PlaceOrderIntent]:
        shares = self.account_view.shares
        if shares <= 0:
            return None
        
        mid_price = self._estimate_price(view)
        if mid_price is None:
            return None
        
        panic_sell_price = mid_price - (5 * self.spread_size)
        panic_sell_price = max(0.01, panic_sell_price)
        quantity = min(shares, self.order_size * 3)# Larger size in panic
        
        self.last_mid_price = mid_price
        
        return PlaceOrderIntent(
            intent_id=self.intent_id,
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=panic_sell_price
        )
    
    
    def _create_panic_buy_intent(self, view: AgentView) -> Optional[PlaceOrderIntent]:
        cash = self.account_view.cash
        if cash <= 0:
            return None
        
        mid_price = self._estimate_price(view)
        if mid_price is None:
            return None
        
        panic_buy_price = mid_price + (5 * self.spread_size)
        
        # Calculate affordable quantity
        cost_per_share = panic_buy_price * (1 + self.constants.fee_rate)
        max_quantity = int(cash / cost_per_share)
        quantity = min(max_quantity, self.order_size * 3)
        
        if quantity <= 0:
            return None
        
        self.last_mid_price = mid_price
        
        return PlaceOrderIntent(
            intent_id=self.intent_id,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=panic_buy_price
        )
    
        
    def _should_stabilize(self) -> bool:
        inventory_ratio = self._calculate_inventory_ratio()
        deviation = abs(inventory_ratio - self.target_inventory_fraction)
        
        return deviation > self.stabilization_tolerance
    
    
    def _stabilize_layer(self, view: AgentView) -> List[AgentIntent]:
        intents = []
        
        inventory_ratio = self._calculate_inventory_ratio()
        skew = self._calculate_skew(inventory_ratio)
        
        mid_price = self._estimate_price(view)
        if mid_price is None:
            return intents
        
        
        bid_price, ask_price = self._calculate_skewed_prices(mid_price, skew)
        bid_quantity, ask_quantity = self._calculate_skewed_quantities(inventory_ratio)
        
        if bid_price >= ask_price:
            # Ensure bid < ask
            spread = max(1.0, self.spread_size / 2.0)
            bid_price = mid_price - spread
            ask_price = mid_price + spread
        
        
        intents.extend(self._selective_stabilize_cancel(inventory_ratio, bid_price, ask_price))
        
        # Check existing coverage after selective cancel
        has_equivalent_bid = self._has_equivalent_working_order(Side.BUY, bid_price)
        has_equivalent_ask = self._has_equivalent_working_order(Side.SELL, ask_price)
        
        if not has_equivalent_bid and bid_quantity > 0 and bid_price > 0:
            intents.append(PlaceOrderIntent(
                intent_id=self.intent_id,
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=bid_quantity,
                price=bid_price
            ))
        
        if not has_equivalent_ask and ask_quantity > 0 and ask_price > 0:
            intents.append(PlaceOrderIntent(
                intent_id=self.intent_id,
                side=Side.SELL,
                order_type=OrderType.LIMIT,
                quantity=ask_quantity,
                price=ask_price
            ))
        
        self.last_mid_price = mid_price
        
        return intents
    
    
    def _calculate_skew(self, inventory_ratio: float) -> float:
        #Calculate price skew based on inventory imbalance.
        #Positive skew = need to buy (shift quotes up)
        #Negative skew = need to sell (shift quotes down)

        deviation = inventory_ratio - self.target_inventory_fraction
        skew = -deviation * self.skew_factor
        return skew
    
    
    def _calculate_skewed_prices(self, mid_price: float, skew: float) -> Tuple[float, float]:
        max_shift = self.spread_size / 2.0  # ±1 tick absolute

        skew_direction = 1 if skew > 0 else -1 if skew < 0 else 0
        price_shift = skew_direction * min(abs(skew / 100.0 * mid_price), max_shift)
        
        # Apply limited shift to mid point
        shifted_mid = mid_price + price_shift
        
        # Calculate base spread
        half_spread = self.spread_size / 2.0
        
        bid_price = shifted_mid - half_spread
        ask_price = shifted_mid + half_spread
        
        fee_per_share = mid_price * self.constants.fee_rate
        bid_price = bid_price - fee_per_share
        ask_price = ask_price + fee_per_share
        
        return max(0.01, bid_price), max(0.01, ask_price)
    
    
    def _calculate_skewed_quantities(self, inventory_ratio: float) -> Tuple[int, int]:
        base_quantity = self.order_size
        
        # Calculate deviation magnitude for proportional skew
        deviation = abs(inventory_ratio - self.target_inventory_fraction)
        
        # Scale skew factor: larger deviation = more aggressive skew
        # Base: 2x supportive, 0.5x risk-creating
        supportive_multiplier = min(2.0 + deviation * 2, 3.0)  # Cap at 3x
        risk_multiplier = max(0.5 - deviation, 0.25)  # Floor at 0.25x
        
        if inventory_ratio > self.target_inventory_fraction:
            # Too much cash - bigger bid (supportive), smaller ask (risk)
            bid_quantity = int(base_quantity * supportive_multiplier)
            ask_quantity = int(base_quantity * risk_multiplier)
        else:
            # Too many shares - smaller bid (risk), bigger ask (supportive)
            bid_quantity = int(base_quantity * risk_multiplier)
            ask_quantity = int(base_quantity * supportive_multiplier)
        
        bid_quantity = self._validate_bid_quantity(bid_quantity)
        ask_quantity = self._validate_ask_quantity(ask_quantity)
        
        return bid_quantity, ask_quantity
    
        
    def _provide_layer(self, view: AgentView) -> List[AgentIntent]:
        intents = []
        
        mid_price = self._estimate_price(view)
        if mid_price is None:
            return intents
        
        bid_price, ask_price = self._calculate_provide_prices(mid_price)
        
        # Ensure wash trade prevention
        if bid_price >= ask_price:
            return intents
        
        #Check if we need to refresh
        if not self._should_refresh_quotes(mid_price):
            return []
        
        #Selective cancel
        intents.extend(self._cancel_non_equivalent_orders(bid_price, ask_price))
        
        #Check existing coverage
        has_equivalent_bid = self._has_equivalent_working_order(Side.BUY, bid_price)
        has_equivalent_ask = self._has_equivalent_working_order(Side.SELL, ask_price)
        
        if not has_equivalent_bid:
            bid_quantity = self._validate_bid_quantity(self.order_size)
            if bid_quantity > 0:
                intents.append(PlaceOrderIntent(
                    intent_id=self.intent_id,
                    side=Side.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=bid_quantity,
                    price=bid_price
                ))
        
        if not has_equivalent_ask:
            ask_quantity = self._validate_ask_quantity(self.order_size)
            if ask_quantity > 0:
                intents.append(PlaceOrderIntent(
                    intent_id=self.intent_id,
                    side=Side.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=ask_quantity,
                    price=ask_price
                ))
        
        if intents:
            self.last_quote_global_tick = self.current_global_tick
            self.last_mid_price = mid_price
        
        return intents
    
    
    def _calculate_provide_prices(self, mid_price: float) -> Tuple[float, float]:
        half_spread = self.spread_size / 2.0
        
        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread
        
        total_fee_per_share = 2 * self.constants.fee_rate * mid_price
        
        bid_price = bid_price - total_fee_per_share / 2.0
        ask_price = ask_price + total_fee_per_share / 2.0
        
        return max(0.01, bid_price), max(0.01, ask_price)
    
    
    def _should_refresh_quotes(self, current_mid_price: Optional[float] = None) -> bool:
        #- Order DONE (filled/cancelled)
        #- Mid-price drift > tolerance
        #- Layer transition / inventory direction change
        #- No working orders

        if self.last_quote_global_tick is not None:
            elapsed = self.current_global_tick - self.last_quote_global_tick
            if elapsed < self.wait_time:
                return False
        
        if self.last_quote_global_tick is None:
            return True
        
        if self.order_filled_this_tick:
            return True
        
        if self.layer_changed:
            return True
        
        if current_mid_price is not None and self.last_mid_price is not None:
            drift = abs(current_mid_price - self.last_mid_price) / self.last_mid_price if self.last_mid_price > 0 else 0
            drift_tolerance = self.spread_size / (2 * self.last_mid_price) if self.last_mid_price > 0 else 0.01
            if drift > drift_tolerance:
                return True
        
        has_working_bid = any(o.lifecycle == OrderLifecycle.WORKING for o in self.active_bids.values())
        has_working_ask = any(o.lifecycle == OrderLifecycle.WORKING for o in self.active_asks.values())
        
        if not has_working_bid or not has_working_ask:
            return True
        
        return False
    
    
    def _cancel_all_orders(self) -> List[CancelOrderIntent]:
        intents = []
        
        for order_id, order_view in list(self.active_bids.items()):
            if order_view.lifecycle == OrderLifecycle.WORKING:
                intents.append(CancelOrderIntent(
                    intent_id=self.intent_id,
                    order_id=order_id
                ))
        
        for order_id, order_view in list(self.active_asks.items()):
            if order_view.lifecycle == OrderLifecycle.WORKING:
                intents.append(CancelOrderIntent(
                    intent_id=self.intent_id,
                    order_id=order_id
                ))
        
        return intents
        
    def _is_quote_equivalent(self, order_view: OrderView, target_price: float, target_side: Side) -> bool:
        if order_view.lifecycle != OrderLifecycle.WORKING:
            return False
        if order_view.side != target_side:
            return False

        price_tolerance = 1.0
        if order_view.price is not None and abs(order_view.price - target_price) <= price_tolerance:
            return True
        
        return False
    
    
    def _has_equivalent_working_order(self, side: Side, target_price: float) -> bool:
        orders = self.active_bids if side == Side.BUY else self.active_asks
        for order_view in orders.values():
            if self._is_quote_equivalent(order_view, target_price, side):
                return True
        return False
    
    
    def _cancel_non_equivalent_orders(self, bid_price: float, ask_price: float) -> List[CancelOrderIntent]:
        intents = []
        
        for order_id, order_view in list(self.active_bids.items()):
            if order_view.lifecycle == OrderLifecycle.WORKING:
                if not self._is_quote_equivalent(order_view, bid_price, Side.BUY):
                    intents.append(CancelOrderIntent(
                        intent_id=self.intent_id,
                        order_id=order_id
                    ))
        
        for order_id, order_view in list(self.active_asks.items()):
            if order_view.lifecycle == OrderLifecycle.WORKING:
                if not self._is_quote_equivalent(order_view, ask_price, Side.SELL):
                    intents.append(CancelOrderIntent(
                        intent_id=self.intent_id,
                        order_id=order_id
                    ))
        
        return intents
    
    
    def _selective_stabilize_cancel(self, inventory_ratio: float, target_bid: float, target_ask: float) -> List[CancelOrderIntent]:
        #If inventory_ratio < target (too many shares):
        #    - We need to SELL → asks are supportive → NEVER cancel asks
        #    - Cancel non-equivalent BIDS only
        
        #If inventory_ratio > target (too much cash):
        #    - We need to BUY → bids are supportive → NEVER cancel bids
        #    - Cancel non-equivalent ASKS only

        intents = []
        
        if inventory_ratio < self.target_inventory_fraction:
            for order_id, order_view in list(self.active_bids.items()):
                if order_view.lifecycle == OrderLifecycle.WORKING:
                    if not self._is_quote_equivalent(order_view, target_bid, Side.BUY):
                        intents.append(CancelOrderIntent(
                            intent_id=self.intent_id,
                            order_id=order_id
                        ))
        else:
            for order_id, order_view in list(self.active_asks.items()):
                if order_view.lifecycle == OrderLifecycle.WORKING:
                    if not self._is_quote_equivalent(order_view, target_ask, Side.SELL):
                        intents.append(CancelOrderIntent(
                            intent_id=self.intent_id,
                            order_id=order_id
                        ))
        
        return intents
    
    
    def _estimate_price(self, view: AgentView) -> Optional[float]:
        md = view.market_data_view
        assert md is not None
        
        if md.mid_price is not None:
            self.last_known_price = md.mid_price
            return md.mid_price
        
        if md.micro_price is not None:
            self.last_known_price = md.micro_price
            return md.micro_price
        
        if md.last_traded_price is not None:
            self.last_known_price = md.last_traded_price
            return md.last_traded_price

        if self.last_known_price is not None:
            return self.last_known_price

        assert view.economy_insight_view is not None
        tv_l, tv_u = view.economy_insight_view.tv_interval
        
        return (tv_l + tv_u) / 2
    
    
    def _calculate_inventory_ratio(self) -> float:
        #0.0 = all shares, 0.5 = balanced, 1.0 = all cash
        
        cash = self.account_view.cash
        shares = self.account_view.shares
        
        if self.last_known_price is None:
            return self.target_inventory_fraction
        
        total_wealth = cash + shares * self.last_known_price
        
        if total_wealth <= 0:
            return self.target_inventory_fraction
        
        return cash / total_wealth
    
    
    def _validate_bid_quantity(self, quantity: int) -> int:
        if quantity <= 0:
            return 0
        
        cash = self.account_view.cash
        if cash <= 0:
            return 0
        
        if self.last_known_price is None:
            return 0
        
        cost_per_share = self.last_known_price * (1 + self.constants.fee_rate)
        max_quantity = int(cash / cost_per_share)
        
        return min(quantity, max_quantity)
    
    
    def _validate_ask_quantity(self, quantity: int) -> int:
        if quantity <= 0:
            return 0
        
        shares = self.account_view.shares
        return min(quantity, shares)
    
    
    def _cleanup_done_orders(self) -> None:
        for order_id, order_view in list(self.active_bids.items()):
            if order_view.lifecycle == OrderLifecycle.DONE:
                if order_view.end_reason == OrderEndReasons.FILLED:
                    self.order_filled_this_tick = True
                del self.active_bids[order_id]
        

        for order_id, order_view in list(self.active_asks.items()):
            if order_view.lifecycle == OrderLifecycle.DONE:
                if order_view.end_reason == OrderEndReasons.FILLED:
                    self.order_filled_this_tick = True
                del self.active_asks[order_id]
