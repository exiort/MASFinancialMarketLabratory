from __future__ import annotations
from dataclasses import dataclass



@dataclass(frozen=True)
class DepositView:
    deposit_id:int
    agent_id:int

    creation_macro_tick:int
    maturity_macro_tick:int

    deposited_cash:float
    interest_rate:float

    matured_cash:float
