###############################################################################
#
# File:      schemas.py
# Author(s): Nico
# Scope:     Pydantic schemas for JSON config file validation
#
# Created:   23 November 2024
#
###############################################################################
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator
from typing import Literal
from utils.defines import FEATURES_COMPUTATION_TYPES, MODELS_TYPES, EVAL_METRICS


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
    features_computation: str = Field(
        description="Represents how the initial population will be initialized - by setting random ones per row "
                    "('rows', i.e. by individual), or per column ('columns', i.e. by feature)",
        default="columns",
    )
    ml_model: str = Field(
        description="Define which type of ML model is used for model definition - available ones are 'ET' (sklearn "
                    "ExtraTreesClassifier)",
        default="ET"
    )
    eval_metric: str = Field(
        description="Define which evaluation metric to use for all the ML models - currently only single objective is "
                    "supported - available ones are 'F1' (F1-score)",
        default="F1"
    )
    random_seed: int = Field(
        description="Seed for reproducibility",
        default=42
    )

    @field_validator("features_computation")
    def validate_features_computation(cls, v):
        if v not in FEATURES_COMPUTATION_TYPES:
            raise ValueError(f"features_computation must be one of {FEATURES_COMPUTATION_TYPES} - provided '{v}'")
        return v

    @field_validator("ml_model")
    def validate_ml_model(cls, v):
        if v not in MODELS_TYPES:
            raise ValueError(f"ml_model must be one of {MODELS_TYPES} - provided '{v}'")
        return v

    @field_validator("eval_metric")
    def validate_eval_metric(cls, v):
        if v not in EVAL_METRICS:
            raise ValueError(f"eval_metric must be one of {EVAL_METRICS} - provided '{v}'")
        return v
