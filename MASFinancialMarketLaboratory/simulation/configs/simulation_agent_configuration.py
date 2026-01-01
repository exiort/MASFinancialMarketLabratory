from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict



class DistributionConfig(BaseModel):
    distribution: str
    model_config = {"frozen": True, "extra": "allow"}

    
class InitialEndowment(BaseModel):
    cash: DistributionConfig
    shares: DistributionConfig
    model_config = {"frozen": True}

    
class InitialEndowments(BaseModel):
    market_maker: InitialEndowment
    others: InitialEndowment
    model_config = {"frozen": True}

    
class AgentGroup(BaseModel):
    count: int
    parameters: Dict[str, DistributionConfig]
    model_config = {"frozen": True}

    
class GlobalConfig(BaseModel):
    random_seed: int
    model_config = {"frozen": True}

    
class SimulationAgentConfiguration(BaseSettings):
    global_config: GlobalConfig = Field(alias="global")
    initial_endowment: InitialEndowments
    
    market_maker: AgentGroup
    value_investor: AgentGroup
    momentum_trader: AgentGroup
    noise_trader: AgentGroup

    model_config = SettingsConfigDict(frozen=True)

    
__SIMULATION_AGENT_CONFIGURATION: Optional[SimulationAgentConfiguration] = None


def set_simulation_agent_configuration(config: SimulationAgentConfiguration) -> None:
    global __SIMULATION_AGENT_CONFIGURATION
    __SIMULATION_AGENT_CONFIGURATION = config

    
def get_simulation_agent_configuration() -> SimulationAgentConfiguration:
    assert __SIMULATION_AGENT_CONFIGURATION is not None
    return __SIMULATION_AGENT_CONFIGURATION
