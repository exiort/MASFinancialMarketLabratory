from __future__ import annotations
from dataclasses import dataclass

from environment.views import DepositView
from environment.configs import get_environment_configuration



@dataclass(frozen=True)
class Deposit:
    deposit_id:int
    agent_id:int

    timestamp:float
    creation_macro_tick:int
    maturity_macro_tick:int
    
    deposited_cash:int
    interest_rate:float

    matured_cash:int


    def create_view(self) -> DepositView:
        ENV_CONFIG = get_environment_configuration()
        
        return DepositView(
            deposit_id=self.deposit_id,
            agent_id=self.agent_id,
            creation_macro_tick=self.creation_macro_tick,
            maturity_macro_tick=self.maturity_macro_tick,
            deposited_cash=self.deposited_cash / ENV_CONFIG.PRICE_SCALE,
            interest_rate=self.interest_rate,
            matured_cash=self.matured_cash / ENV_CONFIG.PRICE_SCALE
        )
