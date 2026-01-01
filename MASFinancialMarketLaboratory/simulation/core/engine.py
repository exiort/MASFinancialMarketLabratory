from __future__ import annotations
from typing import List, Tuple, Optional
import time

from environment import Environment
from environment.views import AccountView
from agents import Agent, ValueInvestor, MarketMaker
from agents.models import AgentView, AgentFeedback
from agents.intents import AgentIntent, PlaceOrderIntent, CancelOrderIntent, CreateDepositIntent

from .agent_manager import AgentManager
from simulation.configs import get_simulation_realtime_data



class SimulationEngine:
    env: Environment
    agent_manager: AgentManager
    
    
    def __init__(self, env: Environment):
        self.env = env
        
        self.agent_manager = AgentManager(self.__register_agent_callback)
        self.agent_manager.initialize_agents()

        
    def __register_agent_callback(self, agent_id: int, cash: float, shares: int) -> Optional[AccountView]:
        return self.env.register_agent(agent_id, cash, shares)        


    def run(self) -> None:
        SIM_REALTIME_DATA = get_simulation_realtime_data()
        
        while True:
            current_macro = SIM_REALTIME_DATA.MACRO_TICK
            current_micro = SIM_REALTIME_DATA.MICRO_TICK

            print(f"\r\tSimulation running on Macro Tick - {current_macro}, Micro Tick - {current_micro}", end="")
            
            if current_micro == 0:
                self.env.expire_session()
                self._update_views_macro()
                
                agents = self.agent_manager.macro_tick_agents()
                self._agent_loop(agents, current_macro, current_micro)
            
            else:
                self._update_views_micro()
                
                agents = self.agent_manager.micro_tick_agents()
                self._agent_loop(agents, current_macro, current_micro)

            if not SIM_REALTIME_DATA.step_hybrid_time():
                break
    
    
    def _update_views_macro(self) -> None:
        sim_data = get_simulation_realtime_data()
    
        sim_data.set_economy_insight(self.env.get_economy_insight())
        sim_data.set_market_data_view(self.env.get_market_data())
    
    
    def _update_views_micro(self) -> None:
        sim_data = get_simulation_realtime_data()
        
        sim_data.set_market_data_view(self.env.get_market_data())
    
    
    def _agent_loop(
        self, 
        agents: List[Tuple[int, Agent]], 
        macro: int, 
        micro: int
    ) -> None:
        sim_data = get_simulation_realtime_data()
        
        for agent_id, agent in agents:
            economy_view = None
            if isinstance(agent, (ValueInvestor, MarketMaker)):
                economy_view = sim_data.ECONOMY_INSIGHT_VIEW

            view = AgentView(
                agent_id=agent_id,
                timestamp=time.time(),
                macro_tick=macro,
                micro_tick=micro,
                market_data_view=sim_data.MARKET_DATA_VIEW,
                economy_insight_view=economy_view
            )
            
            intents = agent.decide(view)
            if not intents:
                continue

            if not isinstance(agent, ValueInvestor):
                assert all(not isinstance(intent, CreateDepositIntent) for intent in intents)
            
            feedback = self._process_intents(agent_id, intents)
            agent.update(feedback)
    
    
    def _process_intents(
        self, 
        agent_id: int, 
        intents: List[AgentIntent]
    ) -> AgentFeedback:
        order_results = {}
        deposit_results = {}
        
        for intent in intents:
            if isinstance(intent, PlaceOrderIntent):
                order_view = self.env.create_order(
                    agent_id=agent_id,
                    order_type=intent.order_type,
                    side=intent.side,
                    quantity=intent.quantity,
                    price=intent.price
                )
                order_results[intent.intent_id] = order_view
            
            elif isinstance(intent, CancelOrderIntent):
                self.env.cancel_order(agent_id, intent.order_id)
                order_results[intent.intent_id] = None
            
            elif isinstance(intent, CreateDepositIntent):
                deposit_view = self.env.create_deposit(
                    agent_id=agent_id,
                    term=intent.term,
                    deposited_cash=intent.amount
                )
                deposit_results[intent.intent_id] = deposit_view
        
        return AgentFeedback(
            agent_id=agent_id,
            order_results=order_results,
            deposit_results=deposit_results
        )
