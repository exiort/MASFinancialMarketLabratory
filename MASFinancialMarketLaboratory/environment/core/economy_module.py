from __future__ import annotations
from typing import List, Tuple, Optional
import random

from environment.models import EconomyInsight
from environment.configs import get_environment_configuration
from environment.configs.models.economy_scenario import EconomyScenario

from simulation.configs import get_simulation_realtime_data



class EconomyModule:
    scenario:EconomyScenario
    rng:random.Random

    __tv:List[float]
    __r:List[float]
    __width:List[float]

    __interval_low:List[Optional[float]]
    __interval_high:List[Optional[float]]

    __max_generated_tick:int


    def __init__(self) -> None:
        ENV_CONFIG = get_environment_configuration()
        
        self.scenario = ENV_CONFIG.ECONOMY_SCENARIO
        self.rng = random.Random(self.scenario.seed)

        self.__tv = [self.scenario.tv_initial]
        self.__r = [self.scenario.r_initial]

        self.__width = [self.__sample_width()]
        self.__interval_low = [None]
        self.__interval_high = [None]
        
        self.__max_generated_tick = 0


    def __step_tv(self, tv_t:float) -> float:
        eps = self.rng.gauss(0.0, 1.0)

        return (
            tv_t
            + self.scenario.tv_mean_reversion * (self.scenario.tv_long_run_mean - tv_t)
            + self.scenario.tv_drift
            + self.scenario.tv_vol * eps
        )


    def __step_r(self, r_t:float) -> float:
        eps = self.rng.gauss(0.0, 1.0)
        r_next = (
            r_t
            + self.scenario.r_mean_reversion * (self.scenario.r_long_run_mean - r_t)
            + self.scenario.r_vol * eps
        )
        
        return max(1e-8, r_next)


    def __sample_width(self) -> float:
        eps = self.rng.gauss(0.0, 1.0)
        raw = self.scenario.tv_interval_base_width + self.scenario.tv_interval_vol * eps

        return max(1e-8, raw)

    
    def step(self, macro_tick:int) -> None:
        if macro_tick <= self.__max_generated_tick: return

        for t in range(self.__max_generated_tick, macro_tick):
            tv_t = self.__tv[t]
            r_t = self.__r[t]

            tv_next = self.__step_tv(tv_t)
            r_next = self.__step_r(r_t)
            w_next = self.__sample_width()
            
            self.__tv.append(tv_next)
            self.__r.append(r_next)
            self.__width.append(w_next)
            self.__interval_low.append(None)
            self.__interval_high.append(None)
            
        self.__max_generated_tick = macro_tick


    def get_true_value(self, macro_tick:int) -> float:
        self.step(macro_tick)

        return self.__tv[macro_tick]

    
    def get_short_rate(self, macro_tick:int) -> float:
        self.step(macro_tick)

        return self.__r[macro_tick]

    
    def get_width(self, macro_tick:int) -> float:
        self.step(macro_tick)

        return self.__width[macro_tick]

    
    def get_tv_interval(self, macro_tick:int) -> Tuple[float, float]:        
        #Z_t ~ Uniform(0,1)
        #L_t = TV_t - Z_t W_t
        #U_t = TV_t + (1 - Z_t) W_t

        self.step(macro_tick)

        if (
            self.__interval_low[macro_tick] is not None
            and self.__interval_high[macro_tick] is not None
        ):
            return self.__interval_low[macro_tick], self.__interval_high[macro_tick] #type:ignore[arg-type]

        tv = self.__tv[macro_tick]
        width = self.__width[macro_tick]

        z = self.rng.random()

        lower = tv - z * width
        upper = tv + (1.0 - z) * width

        self.__interval_low[macro_tick] = lower
        self.__interval_high[macro_tick] = upper
        
        return lower, upper


    def get_deposit_rates(self, macro_tick:int) -> Tuple[float, ...]:
        self.step(macro_tick)
        r_t = self.__r[macro_tick]
        s = self.scenario

        return tuple(max(0.0, r_t + s.term_curve_slope * x + s.term_curve_curvature * x * x) for x in s.deposit_terms) 


    def get_economy_insight(self) -> EconomyInsight:
        ENV_CONFIG = get_environment_configuration()
        SIM_REALTIME_DATA = get_simulation_realtime_data()
        
        tv = self.get_true_value(SIM_REALTIME_DATA.MACRO_TICK)
        short_rate = self.get_short_rate(SIM_REALTIME_DATA.MACRO_TICK)
        width = self.get_width(SIM_REALTIME_DATA.MACRO_TICK)
        tv_low, tv_high = self.get_tv_interval(SIM_REALTIME_DATA.MACRO_TICK)
        deposit_rates = self.get_deposit_rates(SIM_REALTIME_DATA.MACRO_TICK)

        return EconomyInsight(
            macro_tick=SIM_REALTIME_DATA.MACRO_TICK,
            true_value=int(tv * ENV_CONFIG.PRICE_SCALE),
            short_rate=short_rate,
            width=width,
            tv_interval=(int(tv_low * ENV_CONFIG.PRICE_SCALE), int(tv_high * ENV_CONFIG.PRICE_SCALE)),
            deposit_rates=dict(zip(self.scenario.deposit_terms, deposit_rates))
        )
