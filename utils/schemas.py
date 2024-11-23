###############################################################################
#
# File:      schemas.py
# Author(s): Nico
# Scope:     Pydantic schemas for JSON config file validation
#
# Created:   23 November 2024
#
###############################################################################
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class InputConfig(BaseModel):
    """
    Input config parameters for the genetic algo
    """
    # Forbid extra so we know what's in use in the config
    model_config = ConfigDict(extra='forbid')

    nb_generation: PositiveInt = Field(
        description="Number of generations to run the genetic algorithm",
        default=10
    )
    nb_population: PositiveInt = Field(
        description="Number of population to run the genetic algorithm",
        default=50
    )
