from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional, Tuple
import random
import math
import time

from environment.views import AccountView
from environment.configs import get_environment_configuration
from agents import Agent, MarketMaker, ValueInvestor, MomentumTrader, NoiseTrader
from agents.models import AgentConstants

from simulation.configs import get_simulation_configurations
from simulation.configs.simulation_agent_configuration import get_simulation_agent_configuration, DistributionConfig



class AgentManager:
    register_agent_callback:Callable[[int, float, int], Optional[AccountView]]
    
    rng:random.Random
    
    agents: Dict[int, Agent]
    market_makers: Dict[int, MarketMaker]
    value_investors: Dict[int, ValueInvestor]
    momentum_traders: Dict[int, MomentumTrader]
    noise_traders: Dict[int, NoiseTrader]
    
    
    def __init__(
        self, 
        register_agent_callback: Callable[[int, float, int], Optional[AccountView]]
    ) -> None:
        self.register_agent_callback = register_agent_callback
        
        self.agents = {}
        self.market_makers = {}
        self.value_investors = {}
        self.momentum_traders = {}
        self.noise_traders = {}
        
        agent_config = get_simulation_agent_configuration()
        self.rng = random.Random(agent_config.global_config.random_seed)
    
    
    def initialize_agents(self) -> None:
        print("\tInitializing Agents...", end="", flush=True)
        time.sleep(2)
        self._initialize_market_makers()
        self._initialize_value_investors()
        self._initialize_momentum_traders()
        self._initialize_noise_traders()
        print("\b\b\b - Completed ✔️")
        time.sleep(1)
        
        print(f"\t\tMarketMaker Agents Initialized: {len(self.market_makers)}")
        print(f"\t\tValueInvestor Agents Initialized: {len(self.value_investors)}") 
        print(f"\t\tMomentumTrader Agents Initialized: {len(self.momentum_traders)}")
        print(f"\t\tNoiseTrader Agents Initialized: {len(self.noise_traders)}")
        print(f"\t\tTotal Agents Initialized: {len(self.agents)}")
    
    
    def _get_agent_constants(self) -> AgentConstants:
        sim_config = get_simulation_configurations()
        env_config = get_environment_configuration()
        
        return AgentConstants(
            simulation_macro_tick=sim_config.SIMULATION_MACRO_TICK,
            simulation_micro_tick=sim_config.SIMULATION_MICRO_TICK,
            fee_rate=float(env_config.FEE_RATE_PPM) / 1_000_000.0
        )
    
    
    def _sample_value(self, dist_config: DistributionConfig) -> Any:
        dist_type = dist_config.distribution
        
        if dist_type == "constant":
            return getattr(dist_config, "value")
        
        elif dist_type == "uniform":
            min_val = getattr(dist_config, "min")
            max_val = getattr(dist_config, "max")
            return self.rng.uniform(min_val, max_val)
        
        elif dist_type == "discrete_uniform":
            values = getattr(dist_config, "values")
            return self.rng.choice(values)
        
        elif dist_type == "lognormal":
            mean = getattr(dist_config, "mean")
            std = getattr(dist_config, "std")
            return self.rng.lognormvariate(math.log(mean), std)
        
        raise ValueError(f"Unknown distribution type: {dist_type}")

    
    def _initialize_market_makers(self) -> None:
        config = get_simulation_agent_configuration()
        group = config.market_maker
        endowment = config.initial_endowment.market_maker
        
        for i in range(group.count):
            agent_id = 10000 + i
            
            init_cash = float(self._sample_value(endowment.cash))
            init_shares = int(self._sample_value(endowment.shares))
            
            account_view = self.register_agent_callback(agent_id, init_cash, init_shares)
            if account_view is None:
                raise RuntimeError(f"Failed to register agent {agent_id}")
            
            params = {k: self._sample_value(v) for k, v in group.parameters.items()}
            
            agent = MarketMaker(
                agent_id=agent_id,
                account_view=account_view,
                constants=self._get_agent_constants(),
                target_inventory_fraction=float(params["target_inventory_fraction"]),
                risk_lower_bound=float(params["risk_lower_bound"]),
                risk_upper_bound=float(params["risk_upper_bound"]),
                stabilization_tolerance=float(params["stabilization_tolerance"]),
                spread_size=int(params["spread_size"]),
                order_size=int(params["order_size"]),
                wait_time=int(params["wait_time"]),
                skew_factor=int(params["skew_factor"])
            )
            
            self.market_makers[agent_id] = agent
            self.agents[agent_id] = agent
    
    
    def _initialize_value_investors(self) -> None:
        config = get_simulation_agent_configuration()
        group = config.value_investor
        endowment = config.initial_endowment.others
        
        for i in range(group.count):
            agent_id = 20000 + i
            
            init_cash = float(self._sample_value(endowment.cash))
            init_shares = int(self._sample_value(endowment.shares))
            
            account_view = self.register_agent_callback(agent_id, init_cash, init_shares)
            if account_view is None:
                raise RuntimeError(f"Failed to register agent {agent_id}")
            
            params = {k: self._sample_value(v) for k, v in group.parameters.items()}
            
            agent = ValueInvestor(
                agent_id=agent_id,
                account_view=account_view,
                constants=self._get_agent_constants(),
                initial_optimism=float(params["initial_optimism"]),
                stubbornness=float(params["stubbornness"]),
                belief_update_rate=float(params["belief_update_rate"]),
                mispricing_threshold=float(params["mispricing_threshold"]),
                max_position_fraction=float(params["max_position_fraction"]),
                deposit_affinity=float(params["deposit_affinity"]),
                deposit_allocation_fraction=float(params["deposit_allocation_fraction"]),
                deposit_horizon_preference=float(params["deposit_horizon_preference"]),
                max_order_size=int(params["max_order_size"]),
                patience_discount=float(params["patience_discount"]),
                patience_premium=float(params["patience_premium"])
            )
            
            self.value_investors[agent_id] = agent
            self.agents[agent_id] = agent
    
    
    def _initialize_momentum_traders(self) -> None:
        config = get_simulation_agent_configuration()
        group = config.momentum_trader
        endowment = config.initial_endowment.others
        
        for i in range(group.count):
            agent_id = 30000 + i
            
            init_cash = float(self._sample_value(endowment.cash))
            init_shares = int(self._sample_value(endowment.shares))
            
            account_view = self.register_agent_callback(agent_id, init_cash, init_shares)
            if account_view is None:
                raise RuntimeError(f"Failed to register agent {agent_id}")
            
            params = {k: self._sample_value(v) for k, v in group.parameters.items()}
            
            agent = MomentumTrader(
                agent_id=agent_id,
                account_view=account_view,
                constants=self._get_agent_constants(),
                momentum_window=int(params["momentum_window"]),
                entry_threshold=float(params["entry_threshold"]),
                exit_threshold=float(params["exit_threshold"]),
                aggressiveness=float(params["aggressiveness"]),
                max_exposure_fraction=float(params["max_exposure_fraction"]),
                directional_bias=float(params["directional_bias"]),
                liquidity_baseline=int(params["liquidity_baseline"])
            )
            
            self.momentum_traders[agent_id] = agent
            self.agents[agent_id] = agent
    
    
    def _initialize_noise_traders(self) -> None:
        config = get_simulation_agent_configuration()
        group = config.noise_trader
        endowment = config.initial_endowment.others
        
        for i in range(group.count):
            agent_id = 40000 + i
            
            init_cash = float(self._sample_value(endowment.cash))
            init_shares = int(self._sample_value(endowment.shares))
            
            account_view = self.register_agent_callback(agent_id, init_cash, init_shares)
            if account_view is None:
                raise RuntimeError(f"Failed to register agent {agent_id}")
            
            params = {k: self._sample_value(v) for k, v in group.parameters.items()}
            
            agent_seed = self.rng.randint(0, 2**32 - 1)
            agent_rng = random.Random(agent_seed)
            
            agent = NoiseTrader(
                agent_id=agent_id,
                account_view=account_view,
                constants=self._get_agent_constants(),
                rng=agent_rng,
                p_trade=float(params["p_trade"]),
                p_buy=float(params["p_buy"]),
                p_market_order=float(params["p_market_order"]),
                min_quantity=int(params["min_quantity"]),
                max_quantity=int(params["max_quantity"]),
                price_offset_ticks=int(params["price_offset_ticks"])
            )
            
            self.noise_traders[agent_id] = agent
            self.agents[agent_id] = agent

            
    def macro_tick_agents(self) -> List[Tuple[int, Agent]]:
        candidates = [(aid, agent) for aid, agent in self.agents.items()]
        self.rng.shuffle(candidates)
        return candidates
    
    
    def micro_tick_agents(self) -> List[Tuple[int, Agent]]:
        candidates = []
        for aid, agent in self.agents.items():
            if aid not in self.value_investors:
                candidates.append((aid, agent))
        
        self.rng.shuffle(candidates)
        return candidates
