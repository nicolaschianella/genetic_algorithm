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
import numpy as np
import random

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
        # Train/validation/test datasets and associated targets
        self.train_features, self.val_features, self.test_features = None, None, None
        self.train_target, self.val_target, self.test_target = None, None, None
        # Population binary indicators
        self.current_population = []

        # Load JSON and validate
        self.load_json(path_to_json=path_to_json)

        # Load dataset
        self.load_dataset(path_to_dataset=path_to_dataset,
                          target=target,
                          remove_columns=remove_columns)

        # Seed
        self.rng = random.Random(self.json_config["random_seed"])
        np.random.seed(self.json_config["random_seed"])

        # Split train/validation
        self.split_train_val_test()


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


    def split_train_val_test(self) -> None:
        """
        Splits randomly the whole dataset in training, validation and test sets, according to 'train_fraction' and
        'val_fraction' specified in the config. Please note that it's taken fully randomly (meaning for time series
        data, it will introduce leak)

        Returns:
            None
        """
        try:
            # Calculate number of train samples and generate train indexes and complementary validation/test indexes
            nb_train_samples = int(self.features.shape[0] * self.json_config["train_fraction"])
            train_indexes = self.rng.sample(range(self.features.shape[0]), nb_train_samples)
            val_test_indexes = [i for i in range(self.features.shape[0]) if i not in train_indexes]

            # Repeat the process for validation and test indexes
            nb_val_samples = int((self.features.shape[0] - nb_train_samples) * self.json_config["val_fraction"])
            val_indexes = self.rng.sample(val_test_indexes, nb_val_samples)
            test_indexes = [i for i in range(self.features.shape[0]) if i not in train_indexes and i not in val_indexes]

            # Assign values for features and target
            self.train_features, self.val_features, self.test_features = (self.features.iloc[train_indexes],
                                                                          self.features.iloc[val_indexes],
                                                                          self.features.iloc[test_indexes])
            self.train_target, self.val_target, self.test_target = (self.target.iloc[train_indexes],
                                                                    self.target.iloc[val_indexes],
                                                                    self.target.iloc[test_indexes])

            logging.info(f"Dataset split in {self.train_features.shape[0]} training samples, "
                         f"{self.val_features.shape[0]} validation samples, and {self.test_features.shape[0]} test "
                         f"samples")

        except Exception as e:
            msg = f"An Exception occurred while splitting data in train/validation/test set - full exception: {e}"
            logging.error(msg)
            raise Exception(msg)


    def initialize_population(self) -> None:
        """
        Initializes the population with max_features features in each individual

        Returns:
            None
        """
        # Initialize matrix with nb_population rows and nb_features columns
        population = np.zeros((self.json_config["nb_population"], self.train_features.shape[1]), dtype=int)

        logging.info(f"Using {self.json_config['features_computation']} mode for initial population creation")

        if self.json_config["features_computation"] == "rows":
            # Compute number of features per individual to use
            nb_features = int(self.json_config["features_fraction"] * self.train_features.shape[1])
            logging.info(f"Initializing population with {nb_features} features per individual")

            # Set nb_features ones in each row
            for i in range(self.json_config["nb_population"]):
                ones_indices = np.random.choice(self.train_features.shape[1], nb_features, replace=False)
                population[i, ones_indices] = 1

        if self.json_config["features_computation"] == "columns":
            nb_ones = int(self.json_config["features_fraction"] * self.json_config["nb_population"])
            logging.info(f"Initializing population with each feature appearing {nb_ones} times along all the "
                         f"individuals")

            # Set nb_ones ones in each column
            for j in range(self.train_features.shape[1]):
                ones_indices = np.random.choice(self.json_config["nb_population"], nb_ones, replace=False)
                population[ones_indices, j] = 1

        logging.info("Finished population initialization")

        self.current_population = population


    def optimize(self) -> None:
        """
        Main method to use the genetic class. Will initialize the population, and then loop over the main components,
        i.e. population evaluation, selection, crossover and mutation, for a total of nb_generation iterations

        Returns:
            None
        """
        # Start with population initialization
        self.initialize_population()

        # Main loop
        for generation in range(self.json_config["nb_generation"]):
            logging.info(f"Generation {generation + 1} / {self.json_config['nb_generation']}")
            ...
