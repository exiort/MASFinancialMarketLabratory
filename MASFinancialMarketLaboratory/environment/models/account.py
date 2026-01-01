from __future__ import annotations
from typing import Dict, Tuple
from dataclasses import dataclass, field

from environment.views import AccountView



@dataclass
class Account:
    account_id:int
    agent_id:int

    cash:int
    shares:int

    reserved_cash:Dict[int, Tuple[int, int]] = field(default_factory=dict) #OrderID -> (quantity, price)
    reserved_shares:Dict[int, int] = field(default_factory=dict) #OrderID -> quantity

    deposited_cash:Dict[int, int] = field(default_factory=dict) #DepositID -> depositted
    

    def get_total_reserved_cash(self) -> int:
        if not self.reserved_cash:
            return 0

        total = 0
        for quantity, price in self.reserved_cash.values():
            total += quantity * price

        return total
        

    def get_total_reserved_shares(self) -> int:
        if not self.reserved_shares:
            return 0

        total = 0
        for quantity in self.reserved_shares.values():
            total += quantity

        return total


    def get_total_deposited_cash(self) -> int:
        if not self.deposited_cash:
            return 0

        total = 0
        for deposited in self.deposited_cash.values():
            total += deposited

        return total            

    
    def create_view(self) -> AccountView:
        return AccountView(self)
