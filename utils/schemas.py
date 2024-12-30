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
from typing import Literal
from utils.defines import FEATURES_COMPUTATION_TYPES, MODELS_TYPES


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
        description="Number of populations to run the genetic algorithm",
        default=50
    )
    train_fraction: float = Field(
        description="Represents the desired train fraction to use for ML models - must be within 0 and 1 (both "
                    "excluded)",
        gt=0,
        lt=1,
        default=0.7
    )
    val_fraction: float = Field(
        description="Represents the desired validation fraction to use for ML models, once the train fraction is "
                    "applied. Can be considered as the genetic 'training' set. The rest of the dataset will go for the "
                    "genetic test set - must be within 0 and 1 (both excluded)",
        gt=0,
        lt=1,
        default=0.7
    )
    features_fraction: float = Field(
        description="Represents the desired feature fraction to use among all the features within the dataset - either "
                    "computed by row (nb_population) or column (nb_features), according to what is set in the "
                    "'features_computation' parameter",
        gt=0,
        lt=1,
        default=0.8
    )
    features_computation: Literal[FEATURES_COMPUTATION_TYPES] = Field(
        description="Represents how the initial population will be initialized - by setting random ones per row "
                    "('rows', i.e. by individual), or per column ('columns', i.e. by feature)",
        default="columns",
    )
    ml_model: Literal[MODELS_TYPES] = Field(
        description="Define which type of ML model is used for model definition - available ones are 'ET' (sklearn "
                    "ExtraTreesClassifier)",
        default="ET"
    )
    random_seed: int = Field(
        description="Seed for reproducibility",
        default=42
    )
