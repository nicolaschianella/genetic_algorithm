###############################################################################
#
# File:      genetic.py
# Author(s): Nico
# Scope:     Main Genetic Algorithm class
#
# Created:   20 November 2024
#
###############################################################################
import json
import logging
import pandas as pd

from utils.defines import TARGET_ENCODER
from utils.schemas import InputConfig


class Genetic:
    """
    Main Genetic class implementation
    """
    def __init__(self,
                 path_to_json: str,
                 path_to_dataset: str,
                 target: str,
                 remove_columns: list[str]) -> None:
        """
        Initialize the Genetic class
        Args:
            path_to_json (str): path to JSON configuration file
            path_to_dataset (str): path to dataset,
            target (str): target column name within the dataset
            remove_columns (list[str]): list of columns to remove from the dataset (to not use them as features)

        Returns
            None
        """
        # Main config dictionary used within the Genetic class
        self.json_config = None
        # Features to test within the dataset (whole data - remove_columns - target)
        self.features = None
        # Target within the dataset
        self.target = None

        # Load JSON and validate
        self.load_json(path_to_json=path_to_json)

        # Load dataset
        self.load_dataset(path_to_dataset=path_to_dataset,
                          target=target,
                          remove_columns=remove_columns)


    def load_json(self,
                  path_to_json: str) -> None:
        """
        Loads JSON configuration file as a class attribute

        Args:
            path_to_json (str): path to JSON configuration file

        Returns:
            None, raises Exception if loading or validation fails
        """
        logging.info(f"Loading JSON configuration file: {path_to_json}")

        try:
            # Read the file
            with open(path_to_json, "r") as f:
                config_file = json.load(f)

            # Validate using Pydantic and convert to dictionary
            self.json_config = dict(InputConfig.model_validate(config_file))

            logging.info(f"Successfully loaded JSON configuration file: {path_to_json}")

        except Exception as e:
            msg = f"Failed to load JSON configuration file: {path_to_json}, full exception: {e}"
            logging.error(msg)
            raise Exception(msg)


    def load_dataset(self,
                     path_to_dataset: str,
                     target: str,
                     remove_columns: list[str]) -> None:
        """
        Loads provided dataset, removes desired columns, checks if target is inside, assigns class attributes

        Args:
            path_to_dataset (str): path to dataset to load
            target (str): target column name within the dataset
            remove_columns (list[str]): list of columns to remove from the dataset (to not use them as features)

        Returns:
            None, raises Exception if loading fails, columns couldn't be removed, or target is missing
        """
        msg = f"Loading dataset: {path_to_dataset}, with target: {target}"

        if remove_columns:
            msg += f", and removing columns: {remove_columns}"

        logging.info(msg)

        try:
            # Read dataset
            data = pd.read_csv(path_to_dataset)
            # Remove columns
            if remove_columns:
                data = data.drop(remove_columns, axis=1)
            # We drop columns that are entirely composed of NaNs
            # Note: in this project, no cleaning will be done, so I just drop every NaN here and assume all data can
            # be used as it is in an ML model
            data = data.dropna(axis=1).dropna()
            # Remove target from dataset, keep it in another class attribute, encode it, and get features as well
            self.target = data[target].map(TARGET_ENCODER)
            self.features = data.drop(target, axis=1)

            msg = f"Successfully loaded dataset: {path_to_dataset}, with target: {target}"

            if remove_columns:
                msg += f", and removed columns: {remove_columns}"

            msg += f". The dataset is now composed of {len(self.features.columns)} features"

            logging.info(msg)

        except Exception as e:
            msg = f"An Exception occurred while loading dataset: {path_to_dataset}, with target: {target}"

            if remove_columns:
                msg += f", and removing columns: {remove_columns}"

            msg += f" - full exception: {e}"
            logging.error(msg)
            raise Exception(msg)


    def optimize(self) -> None:
        ...
