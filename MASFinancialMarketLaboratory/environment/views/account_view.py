from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple

from environment.configs import get_environment_configuration

if TYPE_CHECKING:
    from environment.models import Account



class AccountView:
    def __init__(self, account:Account) -> None:
        self.__account = account

        
    @property
    def account_id(self) -> int:
        return self.__account.account_id


    @property
    def agent_id(self) -> int:
        return self.__account.agent_id


    @property
    def cash(self) -> float:
        ENV_CONFIG = get_environment_configuration()
        return self.__account.cash / ENV_CONFIG.PRICE_SCALE


    @property
    def shares(self) -> int:
        return self.__account.shares


    @property
    def reserved_cash(self) -> Dict[int, Tuple[int, float]]:
        ENV_CONFIG = get_environment_configuration()
        
        return {k: (v[0], v[1] / ENV_CONFIG.PRICE_SCALE) for k, v in self.__account.reserved_cash.items()}


    @property
    def reserved_shares(self) -> Dict[int, int]:
        return self.__account.reserved_shares.copy()


    @property
    def deposited_cash(self) -> Dict[int, float]:
        ENV_CONFIG = get_environment_configuration()

        return {k: v / ENV_CONFIG.PRICE_SCALE for k, v in self.__account.deposited_cash.items()}
        
