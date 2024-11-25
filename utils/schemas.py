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
    train_fraction: float = Field(
        description="Represents the desired train fraction to use for ML models - must be within 0 and 1 (both "
                    "excluded)",
        gt=0,
        lt=1,
        default=0.8
    )
    max_features: PositiveInt = Field(
        description="Maximum number of features to use in ML models",
        default=20
    )
    random_seed: int = Field(
        description="Seed for reproducibility",
        default=42
    )
