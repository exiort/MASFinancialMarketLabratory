from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

from .models import EconomyScenario



class EnvironmentConfiguration(BaseSettings):
    PRICE_SCALE:int
    DB_PATH:str
    
    INSIGHT_L2_DEPTH:int
    ECONOMY_SCENARIO:EconomyScenario

    FEE_RATE_PPM:int #Parts Per Million
    model_config = SettingsConfigDict(frozen=True)


__ENVIRONMENT_CONFIGURATION:Optional[EnvironmentConfiguration] = None


def set_environment_configuration(environment_configuration:EnvironmentConfiguration) -> None:
    global __ENVIRONMENT_CONFIGURATION
    __ENVIRONMENT_CONFIGURATION = environment_configuration


def get_environment_configuration() -> EnvironmentConfiguration:
    assert __ENVIRONMENT_CONFIGURATION is not None

    return __ENVIRONMENT_CONFIGURATION
