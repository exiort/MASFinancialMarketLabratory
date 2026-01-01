from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

from environment.views import EconomyInsightView
from environment.configs import get_environment_configuration



@dataclass(frozen=True)
class EconomyInsight:
    macro_tick:int
    
    true_value:int
    short_rate:float

    width:float

    tv_interval:Tuple[int, int]
    deposit_rates:Dict[int, float] #((term-rate), ...)
    

    def create_view(self) -> EconomyInsightView:
        ENV_CONFIG = get_environment_configuration()
        
        return EconomyInsightView(
            macro_tick=self.macro_tick,
            tv_interval=(self.tv_interval[0] / ENV_CONFIG.PRICE_SCALE, self.tv_interval[1] / ENV_CONFIG.PRICE_SCALE),
            deposit_rates=self.deposit_rates
        )
