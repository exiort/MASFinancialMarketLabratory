from __future__ import annotations
from typing import Optional
import time

from environment.configs import get_environment_configuration
from environment.core import CDAEngine, EconomyModule, SettlementLedger, StorageLedger
from environment.models.order import Order, OrderLifecycle, OrderEndReasons, Side, OrderType
from environment.views import AccountView, DepositView, MarketDataView, OrderView, EconomyInsightView

from simulation.configs import get_simulation_configurations, get_simulation_realtime_data



class Environment:
    settlement_ledger:SettlementLedger
    cda_engine:CDAEngine
    storage_ledger:StorageLedger
    economy_module:EconomyModule
    
    __next_order_id:int
    
    def __init__(self) -> None:      
        self.storage_ledger = StorageLedger()
        self.settlement_ledger = SettlementLedger(self.storage_ledger)
        self.cda_engine = CDAEngine(
            storage_ledger=self.storage_ledger,
            settlement_ledger=self.settlement_ledger
        )

        self.economy_module = EconomyModule()        
        self.__next_order_id = 0
        

    @property
    def order_id(self) -> int:
        order_id = self.__next_order_id
        self.__next_order_id += 1

        return order_id

    
    def register_agent(
            self,
            agent_id:int,
            initial_cash:float=0.0,
            initial_shares:int=0
    ) -> Optional[AccountView]:
        if self.settlement_ledger.is_account_exist(agent_id): return
        if initial_cash < 0: return
        if initial_shares < 0: return

        account = self.settlement_ledger.register_account(
            agent_id=agent_id,
            initial_cash=initial_cash,
            initial_shares=initial_shares
        )

        assert self.storage_ledger.add_account(account)

        return account.create_view()
        
    
    def create_order(
            self,
            agent_id:int,
            order_type:OrderType,
            side:Side,
            quantity:int,
            price:Optional[float]=None
    ) -> Optional[OrderView]:
        if not self.settlement_ledger.is_account_exist(agent_id):
            return

        if not quantity > 0:
            return

        ENV_CONFIG = get_environment_configuration()
        
        if order_type == OrderType.LIMIT:
            if price is None or price <= 0:
                return
            price = int(price * ENV_CONFIG.PRICE_SCALE)
            
        elif order_type == OrderType.MARKET:
            if price is not None:
                return

        SIM_REALTIME_DATA = get_simulation_realtime_data()
        order = Order(
            order_id=self.order_id,
            agent_id=agent_id,
            timestamp=time.time(),
            macro_tick=SIM_REALTIME_DATA.MACRO_TICK,
            micro_tick=SIM_REALTIME_DATA.MICRO_TICK,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            lifecycle=OrderLifecycle.NEW,
            end_reason=OrderEndReasons.NONE
        )

        assert self.storage_ledger.add_order(order)
        
        self.cda_engine.process_new_order(order)
        
        return order.create_view()


    def cancel_order(self, agent_id:int, order_id:int) -> None:
        if not self.settlement_ledger.is_account_exist(agent_id): return

        order = self.storage_ledger.get_order(order_id)
        if order is None: return
        if order.agent_id != agent_id: return
        if order.lifecycle != OrderLifecycle.WORKING: return
        if order.end_reason != OrderEndReasons.NONE: return
        
        self.cda_engine.cancel_order(order_id)
        

    def expire_session(self) -> None:
        self.cda_engine.expire_session()
        self.storage_ledger.flush()


    def create_deposit(
            self,
            agent_id:int,
            term:int,
            deposited_cash:float
    ) -> Optional[DepositView]:
        if not self.settlement_ledger.is_account_exist(agent_id):
            return

        SIM_CONFIG = get_simulation_configurations()
        ENV_CONFIG = get_environment_configuration()
        if not term in ENV_CONFIG.ECONOMY_SCENARIO.deposit_terms:
            return

        SIM_REALTIME_DATA = get_simulation_realtime_data()
        if not SIM_REALTIME_DATA.MACRO_TICK + term <= SIM_CONFIG.SIMULATION_MACRO_TICK:
            return
        
        if not deposited_cash > 0:
            return

        deposit = self.settlement_ledger.create_deposit(
            agent_id=agent_id,
            term=term,
            deposit_cash=deposited_cash,
        )

        if deposit is None: return

        assert self.storage_ledger.add_deposit(deposit)

        return deposit.create_view()

    
    def get_economy_insight(self) -> EconomyInsightView:
        economy_insight = self.economy_module.get_economy_insight()

        assert self.storage_ledger.add_economy_insight(economy_insight)

        return economy_insight.create_view()
        

    def get_market_data(self) -> MarketDataView:
        market_data = self.cda_engine.get_market_data()

        assert self.storage_ledger.add_market_data(market_data)
        
        return market_data.create_view()
