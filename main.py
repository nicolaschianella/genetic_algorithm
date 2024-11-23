###############################################################################
#
# File:      main.py
# Author(s): Nico
# Scope:     Entry point
#
# Created:   20 November 2024
#
###############################################################################
import argparse
import logging
import os

from genetic.genetic import Genetic


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser(description="Genetic Algo")
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        default=os.path.join(os.getcwd(), "config.json"),
        help="Specify path to JSON configuration file",
        required=False
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="store",
        default=os.path.join(os.getcwd(), "data", "breast_cancer.csv"),
        help="Specify path to dataset to load",
        required=False
    )
    parser.add_argument(
        "-t",
        "--target",
        action="store",
        default="diagnosis",
        help="Specify target name within the dataset",
        required=False
    )
    parser.add_argument(
        "-r",
        "--remove_columns",
        action="store",
        nargs="+",
        default=["id"],
        help="Specify columns to remove within the dataset",
        required=False
    )
    parser.add_argument(
        "-l",
        "--log",
        action="store",
        default="genetic.log",
        help="Specify output log file",
        required=False
    )

    args = parser.parse_args()

    # Simple logging basicConfig, appearing in the console as well
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s -- %(filename)s -- %(funcName)s -- %(levelname)s -- %(message)s",
        filemode='w'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s -- %(filename)s -- %(funcName)s -- %(levelname)s -- %(message)s")
    console_handler.setFormatter(formatter)

    # Add the custom handler to the root logger
    logging.getLogger().addHandler(console_handler)

    # Instantiate and run main class instance
    genetic = Genetic(
        path_to_json=args.config,
        path_to_dataset=args.dataset,
        target=args.target,
        remove_columns=args.remove_columns
    )
    genetic.optimize()
