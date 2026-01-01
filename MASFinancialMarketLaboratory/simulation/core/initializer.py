import json
from typing import Any, Dict, List

from environment.configs.environment_configuration import EnvironmentConfiguration, EconomyScenario
from environment.configs import set_environment_configuration
from environment.views.economy_insight_view import EconomyInsightView
from environment.views.market_data_view import MarketDataView

from simulation.configs import set_simulation_configuration, set_simulation_realtime_data
from simulation.configs.simulation_configurations import SimulationConfigurations, get_simulation_configurations 
from simulation.configs.simulation_realtime_data import SimulationRealTimeData
from simulation.configs.simulation_agent_configuration import SimulationAgentConfiguration, set_simulation_agent_configuration


class SimulationInitializer:
    @staticmethod
    def INITIALIZE_CONFIGS(config_json_path:str) -> None:
        with open(config_json_path) as json_file:
            config:Dict[str, Dict[str, Any]] = json.load(json_file)

        environment_config = config.get("environment_config")
        assert environment_config is not None
        SimulationInitializer.INITIALIZE_ENVIRONMENT(environment_config)

        simulation_config = config.get("simulation_config")
        assert simulation_config is not None
        SimulationInitializer.INITIALIZE_SIMULATION(simulation_config)

        agents_config = config.get("agents_config")
        assert agents_config is not None
        SimulationInitializer.INITIALIZE_AGENT_CONFIG(agents_config)

        SimulationInitializer.INITIALIZE_REALTIME_DATA()

        
    @staticmethod
    def INITIALIZE_ENVIRONMENT(environment_config:Dict[str, Any]) -> None:
        seed=environment_config["economy_scenario_seed"]
        assert isinstance(seed, int)
        tv_initial=environment_config["economy_scenario_tv_initial"]
        assert isinstance(tv_initial, float)
        tv_long_run_mean=environment_config["economy_scenario_tv_long_run_mean"]
        assert isinstance(tv_long_run_mean, float)
        tv_drift=environment_config["economy_scenario_tv_drift"]
        assert isinstance(tv_drift, float)
        tv_mean_reversion=environment_config["economy_scenario_tv_mean_reversion"]
        assert isinstance(tv_mean_reversion, float)
        tv_vol=environment_config["economy_scenario_tv_vol"]
        assert isinstance(tv_vol, float)
        r_initial=environment_config["economy_scenario_r_initial"]
        assert isinstance(r_initial, float)
        r_long_run_mean=environment_config["economy_scenario_r_long_run_mean"]
        assert isinstance(r_long_run_mean, float)
        r_mean_reversion=environment_config["economy_scenario_r_mean_reversion"]
        assert isinstance(r_mean_reversion, float)
        r_vol=environment_config["economy_scenario_r_vol"]
        assert isinstance(r_vol, float)
        tv_interval_base_width=environment_config["economy_scenario_tv_interval_base_width"]
        assert isinstance(tv_interval_base_width, float)
        tv_interval_vol=environment_config["economy_scenario_tv_interval_vol"]
        assert isinstance(tv_interval_vol, float)
        term_curve_slope=environment_config["economy_scenario_term_curve_slope"]
        assert isinstance(term_curve_slope, float)
        term_curve_curvature=environment_config["economy_scenario_term_curve_curvature"]
        assert isinstance(term_curve_curvature, float)
        deposit_terms=environment_config["economy_scenario_deposit_terms"]
        assert isinstance(deposit_terms, List)

        scenario = EconomyScenario(
            seed=seed,
            tv_initial=tv_initial,
            tv_long_run_mean=tv_long_run_mean,
            tv_drift=tv_drift,
            tv_mean_reversion=tv_mean_reversion,
            tv_vol=tv_vol,
            r_initial=r_initial,
            r_long_run_mean=r_long_run_mean,
            r_mean_reversion=r_mean_reversion,
            r_vol=r_vol,
            tv_interval_base_width=tv_interval_base_width,
            tv_interval_vol=tv_interval_vol,
            term_curve_slope=term_curve_slope,
            term_curve_curvature=term_curve_curvature,
            deposit_terms=tuple(deposit_terms)
        )

        price_scale = environment_config["price_scale"]
        assert isinstance(price_scale, int)
        db_path = environment_config["db_path"]
        assert isinstance(db_path, str)
        insight_l2_depth = environment_config["insight_l2_depth"]
        assert isinstance(insight_l2_depth, int)
        fee_rate_ppm = environment_config["fee_rate_ppm"]
        assert isinstance(fee_rate_ppm, int)
        
        set_environment_configuration(
            EnvironmentConfiguration(
                PRICE_SCALE=price_scale,
                DB_PATH=db_path,
                INSIGHT_L2_DEPTH=insight_l2_depth,
                ECONOMY_SCENARIO=scenario,
                FEE_RATE_PPM=fee_rate_ppm
            )
        )


    @staticmethod
    def INITIALIZE_SIMULATION(simulation_config:Dict[str, Any]) -> None:
        simulation_macro_tick = simulation_config["simulation_macro_tick"]
        assert isinstance(simulation_macro_tick, int)
        simulation_micro_tick = simulation_config["simulation_micro_tick"]
        assert isinstance(simulation_micro_tick, int)
        init_macro_tick = simulation_config["init_macro_tick"]
        assert isinstance(init_macro_tick, int)
        init_micro_tick = simulation_config["init_micro_tick"]
        assert isinstance(init_micro_tick, int)

        set_simulation_configuration(
            SimulationConfigurations(
                SIMULATION_MACRO_TICK=simulation_macro_tick,
                SIMULATION_MICRO_TICK=simulation_micro_tick,
                INIT_MACRO_TICK=init_macro_tick,
                INIT_MICRO_TICK=init_micro_tick,
            )
        )

    @staticmethod
    def INITIALIZE_AGENT_CONFIG(agent_config:Dict[str, Any]) -> None:
        # Pydantic will parse the dictionary and handle nested models
        # alias="global" allows handling the "global" key
        config = SimulationAgentConfiguration(**agent_config)
        set_simulation_agent_configuration(config)


    @staticmethod
    def INITIALIZE_REALTIME_DATA() -> None:
        SIM_CONFIG = get_simulation_configurations()

        init_econ_insight_view = EconomyInsightView(
            macro_tick=-1,
            tv_interval=(-1, -1),
            deposit_rates={-1: -1}
        )
        init_market_data_view = MarketDataView(
            timestamp=-1,
            macro_tick=-1,
            micro_tick=-1,
            trade_count=-1,
            trade_volume=-1,
            last_traded_price=None,
            last_trade_size=None,
            L1_bids=None,
            L1_asks=None,
            spread=None,
            mid_price=None,
            micro_price=None,
            L2_bids=None,
            L2_asks=None,
            N=-1,
            bids_depth_N=-1,
            asks_depth_N=-1,
            imbalance_N=None,
            vwap_macro=None,
            vwap_micro=None
        )
        
        set_simulation_realtime_data(
            SimulationRealTimeData(
                init_macro_tick=SIM_CONFIG.INIT_MACRO_TICK,
                init_micro_tick=SIM_CONFIG.INIT_MICRO_TICK,
                simulation_macro_tick=SIM_CONFIG.SIMULATION_MACRO_TICK,
                simulation_micro_tick=SIM_CONFIG.SIMULATION_MICRO_TICK,
                economy_insight_view=init_econ_insight_view,
                market_data_view=init_market_data_view
            )
        )
