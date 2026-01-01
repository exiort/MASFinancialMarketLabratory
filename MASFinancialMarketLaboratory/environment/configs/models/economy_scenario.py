from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple



@dataclass(frozen=True)
class EconomyScenario:
    seed:int

    tv_initial:float
    
    tv_long_run_mean:float #μ_TV
    tv_drift:float #δ_TV
    tv_mean_reversion:float #α_TV
    tv_vol:float #σ_TV

    r_initial:float
    r_long_run_mean:float #μ_r
    r_mean_reversion:float #κ_r
    r_vol:float #σ_r

    tv_interval_base_width:float #w_0
    tv_interval_vol:float #σ_w

    term_curve_slope:float #s1
    term_curve_curvature:float #s2

    deposit_terms:Tuple[int, ...]
