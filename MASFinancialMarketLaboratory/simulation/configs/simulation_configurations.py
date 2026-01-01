from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional



class SimulationConfigurations(BaseSettings):
    SIMULATION_MACRO_TICK:int
    SIMULATION_MICRO_TICK:int

    INIT_MACRO_TICK:int
    INIT_MICRO_TICK:int

    model_config = SettingsConfigDict(frozen=True)

    
__SIMULATION_CONFIGURATION:Optional[SimulationConfigurations] = None


def set_simulation_configuration(simulation_configuration:SimulationConfigurations) -> None:
    global __SIMULATION_CONFIGURATION
    __SIMULATION_CONFIGURATION = simulation_configuration


def get_simulation_configurations() -> SimulationConfigurations:
    assert __SIMULATION_CONFIGURATION is not None

    return __SIMULATION_CONFIGURATION
