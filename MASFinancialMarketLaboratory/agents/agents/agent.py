from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from agents.intents import AgentIntent
from agents.models.agent_constants import AgentConstants
from agents.models import AgentView, AgentFeedback

from environment.views.account_view import AccountView



class Agent(ABC):
    agent_id:int
    account_view:AccountView
    constants:AgentConstants
    

    def __init__(self, agent_id:int, account_view:AccountView, constants:AgentConstants) -> None:
        super().__init__()

        self.agent_id = agent_id
        self.account_view = account_view
        self.constants = constants
        
        
    @abstractmethod
    def decide(self, view:AgentView) -> List[AgentIntent]:
        pass


    @abstractmethod
    def update(self, feedback:AgentFeedback) -> None:
        pass
