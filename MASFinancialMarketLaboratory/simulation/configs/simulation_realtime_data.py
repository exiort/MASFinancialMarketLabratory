from typing import Optional

from environment.views import EconomyInsightView, MarketDataView



class SimulationRealTimeData:
    __macro_tick:int
    __micro_tick:int

    __simulation_macro_tick:int
    __simulation_micro_tick:int
    
    __economy_insight_view:EconomyInsightView
    __market_data_view:MarketDataView


    def __init__(
            self,
            init_macro_tick:int,
            init_micro_tick:int,
            simulation_macro_tick:int,
            simulation_micro_tick:int,
            economy_insight_view:EconomyInsightView,
            market_data_view:MarketDataView
    ) -> None:
        self.__macro_tick = init_macro_tick
        self.__micro_tick = init_micro_tick
        self.__simulation_macro_tick = simulation_macro_tick
        self.__simulation_micro_tick = simulation_micro_tick
        self.__economy_insight_view = economy_insight_view
        self.__market_data_view = market_data_view 

        
    @property
    def MACRO_TICK(self) -> int:
        return self.__macro_tick


    @property
    def MICRO_TICK(self) -> int:
        return self.__micro_tick


    @property
    def ECONOMY_INSIGHT_VIEW(self) -> EconomyInsightView:
        assert self.__economy_insight_view.macro_tick == self.__macro_tick

        return self.__economy_insight_view


    @property
    def MARKET_DATA_VIEW(self) -> MarketDataView:
        assert (self.__market_data_view.macro_tick, self.__market_data_view.micro_tick) == (self.__macro_tick, self.__micro_tick)
        
        return self.__market_data_view

    
    def step_hybrid_time(self) -> bool:
        self.__macro_tick += (self.__micro_tick + 1) // self.__simulation_micro_tick
        self.__micro_tick = (self.__micro_tick + 1) % self.__simulation_micro_tick

        if not self.__macro_tick < self.__simulation_macro_tick:
            return False

        return True

    
    def set_economy_insight(self, economy_insight_view:EconomyInsightView) -> None:
        assert economy_insight_view.macro_tick == self.__macro_tick

        self.__economy_insight_view = economy_insight_view


    def set_market_data_view(self, market_data_view:MarketDataView) -> None:
        assert (market_data_view.macro_tick, market_data_view.micro_tick) == (self.__macro_tick, self.__micro_tick)

        self.__market_data_view = market_data_view
        

__SIMULATION_REALTIME_DATA:Optional[SimulationRealTimeData] = None


def set_simulation_realtime_data(simulation_realtime_data:SimulationRealTimeData) -> None:
    global __SIMULATION_REALTIME_DATA
    __SIMULATION_REALTIME_DATA = simulation_realtime_data


def get_simulation_realtime_data() -> SimulationRealTimeData:
    assert __SIMULATION_REALTIME_DATA is not None

    return __SIMULATION_REALTIME_DATA
