#!/usr/bin/python3.7
# type: ignore
########################################################################################
# fitting.py - The parameter-fitting component for the model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################
"""
Used for parameter-fitting of the output of the model runs.

NOTE: The mypy type checker is instructed to ignore this component. This is done due to
the lower standards applied to the analysis code, and the failure of mypy to correctly
type-check the external matplotlib.pyplot module.

"""

import argparse
import datetime
import json
import pdb
import pickle
import re
import sys

from typing import Dict, List, Set, Tuple
import numpy as np  # type: ignore  # pylint: disable=import-error
import pandas as pd  # type: ignore  # pylint: disable=import-error

from dtreeviz.trees import dtreeviz
from matplotlib import pyplot as plt
from scipy.sparse.construct import rand
from scipy.optimize import curve_fit
from scipy.sparse import data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

# Ambient temperature:
#   Keyword for the ambient temperature of the collector.
AMBIENT_TEMPERATURE: str = "ambient_temperature"

# Collector input temperature:
#   Keyword for the input temperature of the collector.
COLLECTOR_INPUT_TEMPERATURE: str = "collector_input_temperature"

# Collector output temperature:
#   Keyword for the output temperature of the collector.
COLLECTOR_OUTPUT_TEMPERATURE: str = "collector_output_temperature"

# Electrical efficiency:
#   Keyword for the electric efficiency of the collector.
ELECTRICAL_EFFICIENCY: str = "electrical_efficiency"

# Mass-flow rate:
#   Keyword for the mass-flow rate of the collector.
MASS_FLOW_RATE: str = "mass_flow_rate"

# Max forest depth:
#   The maximum depth to go to when computing the random forests.
MAX_FOREST_DEPTH: int = 12

# Max tree depth:
#   The maximum depth to go to when computing the individual tree.
MAX_TREE_DEPTH: int = 6

# Number of estimators:
#   The number of estimators ("trees") to include in the random forests.
NUM_ESTIMATORS: int = 100

# Reconstruction resolution:
#   The resolution to use when reconstructing reduced plots.
RECONSTRUCTION_RESOLUTION: int = 800

# Reduced model:
#   Label to use for reduced model data.
REDUCED_MODEL: str = "reduced model"

# Skip resolution:
#   The number of points to skip out when processing the skipped/reduced data for
#   plotting.
SKIP_RESOLUTION: int = 257

# Solar irradiance:
#   Keyword for the solar irradiance.
SOLAR_IRRADIANCE: str = "solar_irradiance"

# Technical model:
#   Label to use for technical 3d model data.
TECHNICAL_MODEL: str = "technical 3d model"

# Thermal efficiency:
#   Keyword for the thermal efficiency.
THERMAL_EFFICIENCY: str = "thermal_efficiency"

# Wind speed:
#   Keyword for the wind speed.
WIND_SPEED: str = "wind_speed"


def _parse_args(args) -> argparse.Namespace:
    """
    Parse the CLI args.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-file-name", "-df", help="Path to the data file to parse."
    )
    parser.add_argument("--use-existing-fits", action="store_true", default=False)

    return parser.parse_args(args)


def analyse(data_file_name: str, use_existing_fits: bool) -> None:
    """
    Analysis function for fitting parameters.

    :param data_file_name:
        The data-file name.
    :param use_existing_fits:
        Whether to use existing fitted data.

    """

    # Parse the input data.
    print("Parsing input data file................ ", end="")
    with open(data_file_name, "r") as f:
        data = json.load(f)
    print("[  DONE  ]")

    # Re-structure the data into a Lasso-friendly format.
    print("Restructuring input data............... ", end="")
    if isinstance(data, list):
        processed_data = pd.DataFrame(
            [
                (
                    entry[AMBIENT_TEMPERATURE],
                    entry[COLLECTOR_INPUT_TEMPERATURE],
                    3600 * entry[MASS_FLOW_RATE],
                    entry[SOLAR_IRRADIANCE],
                    entry[WIND_SPEED],
                    entry[COLLECTOR_OUTPUT_TEMPERATURE],
                    entry[ELECTRICAL_EFFICIENCY],
                )
                for entry in data
                if entry[AMBIENT_TEMPERATURE] is not None
                and entry[COLLECTOR_INPUT_TEMPERATURE] is not None
                and entry[MASS_FLOW_RATE] is not None
                and entry[SOLAR_IRRADIANCE] is not None
                and entry[WIND_SPEED] is not None
                and entry[COLLECTOR_OUTPUT_TEMPERATURE] is not None
                and entry[ELECTRICAL_EFFICIENCY] is not None
            ]
        )
    else:
        raise Exception(
            "Input data must be of type `list`. Other formats are not supported."
        )
    print("[  DONE  ]")

    # Split the last 50 entries out as test data.
    print("Separating out test and train data..... ", end="")
    x_train_therm, x_test_therm, y_train_therm, y_test_therm = train_test_split(
        processed_data[[0, 1, 2, 3, 4]],
        processed_data[5],
        test_size=0.33,
        random_state=42,
    )
    (
        x_train_electric,
        x_test_electric,
        y_train_electric,
        y_test_electric,
    ) = train_test_split(
        processed_data[[0, 1, 2, 3, 4]],
        processed_data[6],
        test_size=0.33,
        random_state=42,
    )

    # Reset the index columns.
    x_train_electric = x_train_electric.reset_index(drop=True)
    x_test_electric = x_test_electric.reset_index(drop=True)
    y_train_therm = y_train_therm.reset_index(drop=True)
    y_test_therm = y_test_therm.reset_index(drop=True)

    print("[  DONE  ]")

    if use_existing_fits:
        with open("electric_tree.sav", "rb") as f:
            electric_tree = pickle.load(f)
        with open("thermal_tree.sav", "rb") as f:
            thermal_tree = pickle.load(f)
        with open("electric_forest.sav", "rb") as f:
            electric_forest = pickle.load(f)
        with open("thermal_forest.sav", "rb") as f:
            thermal_forest = pickle.load(f)

    else:
        # Define the variables needed for the fit.
        electric_tree = DecisionTreeRegressor(
            max_depth=MAX_TREE_DEPTH, min_samples_split=50, min_samples_leaf=10
        )
        thermal_tree = DecisionTreeRegressor(
            max_depth=MAX_TREE_DEPTH, min_samples_split=50, min_samples_leaf=10
        )
        electric_forest = RandomForestRegressor(
            n_estimators=NUM_ESTIMATORS,
            criterion="squared_error",
            max_depth=MAX_FOREST_DEPTH,
            min_samples_split=25,
            min_samples_leaf=5,
        )
        thermal_forest = RandomForestRegressor(
            n_estimators=NUM_ESTIMATORS,
            criterion="squared_error",
            max_depth=MAX_FOREST_DEPTH,
            min_samples_split=25,
            min_samples_leaf=5,
        )

        # Train the models on the data.
        print("Fitting the electrical tree............ ", end="")
        electric_tree.fit(x_train_electric, y_train_electric)
        print("[  DONE  ]")
        print("Fitting the thermal tree............... ", end="")
        thermal_tree.fit(x_train_therm, y_train_therm)
        print("[  DONE  ]")

        print("Fitting the electrical forest.......... ", end="")
        electric_forest.fit(x_train_electric, y_train_electric)
        print("[  DONE  ]")
        print("Fitting the thermal forest............. ", end="")
        thermal_forest.fit(x_train_therm, y_train_therm)
        print("[  DONE  ]")

        # Save the models.
        with open("electric_tree.sav", "wb") as f:
            pickle.dump(electric_tree, f)
        with open("thermal_tree.sav", "wb") as f:
            pickle.dump(thermal_tree, f)

        with open("electric_forest.sav", "wb") as f:
            pickle.dump(electric_forest, f)
        with open("thermal_forest.sav", "wb") as f:
            pickle.dump(thermal_forest, f)

    # Make predictions using these models.
    y_predict_electric_tree = electric_tree.predict(x_test_electric)
    y_predict_therm_tree = thermal_tree.predict(x_test_therm)
    y_predict_electric_forest = electric_forest.predict(x_test_electric)
    y_predict_therm_forest = thermal_forest.predict(x_test_therm)

    # Compute the baseline error and error improvement.
    # The electric baseline error is computed using the collector input temperature as
    # the temperature of the collector.
    electric_error_tree = np.sqrt((y_predict_electric_tree - y_test_electric) ** 2)
    electric_error_tree_baseline = np.sqrt(
        (y_test_electric - 0.125 * (1 - 0.0052 * x_test_electric[1])) ** 2
    )
    electric_error_forest = np.sqrt((y_predict_electric_forest - y_test_electric) ** 2)
    electric_error_forest_baseline = np.sqrt(
        (y_test_electric - 0.125 * (1 - 0.0052 * x_test_electric[1])) ** 2
    )
    # The thermal baseline error is computed using the collector output temperature as
    # the temperature of the collector.
    thermal_error_tree = np.sqrt((y_predict_therm_tree - y_test_therm) ** 2)
    thermal_error_tree_baseline = np.sqrt((y_test_therm - x_test_therm[1]) ** 2)
    thermal_error_forest = np.sqrt((y_predict_therm_forest - y_test_therm) ** 2)
    thermal_error_forest_baseline = np.sqrt((y_test_therm - x_test_therm[1]) ** 2)
    print(
        f"The electric tree had a sd of {100 * np.mean(electric_error_tree): .3g}% efficiency."
    )
    print(
        f"This compares to a baseline electric sd of {100 * np.mean(electric_error_tree_baseline): .3g}% efficiency."
    )
    print(
        f"The electric forest had a sd of {100 * np.mean(electric_error_forest): .3g}% efficiency."
    )
    print(
        f"This compares to a baseline electric sd of {100 * np.mean(electric_error_forest_baseline): .3g}% efficiency."
    )
    print(f"The thermal tree had a sd of {np.mean(thermal_error_tree): .3g}degC.")
    print(
        f"This compares to a baseline thermal sd of {np.mean(thermal_error_tree_baseline): .3g}degC."
    )
    print(f"The thermal forest had a sd of {np.mean(thermal_error_forest): .3g}degC.")
    print(
        f"This compares to a baseline thermal sd of {np.mean(thermal_error_forest_baseline): .3g}degC."
    )

    # Compute the mean average percentage error as a measure of the accuracy.
    electric_tree_mape = 100 * (electric_error_tree / y_test_electric)
    electric_tree_accuracy = 100 - np.mean(electric_tree_mape)
    print(f"The electric tree had an accuracy of {electric_tree_accuracy: .3g}%.")
    electric_forest_mape = 100 * (electric_error_forest / y_test_electric)
    electric_forest_accuracy = 100 - np.mean(electric_forest_mape)
    print(f"The electric forest had an accuracy of {electric_forest_accuracy: .3g}%.")
    thermal_tree_mape = 100 * (thermal_error_tree / y_test_therm)
    thermal_tree_accuracy = 100 - np.mean(thermal_tree_mape)
    print(f"The thermal tree had an accuracy of {thermal_tree_accuracy: .3g}%.")
    thermal_forest_mape = 100 * (thermal_error_forest / y_test_therm)
    thermal_forest_accuracy = 100 - np.mean(thermal_forest_mape)
    print(f"The thermal forest had an accuracy of {thermal_forest_accuracy: .3g}%.")

    x_test_electric_skipped = pd.DataFrame(
        x_test_electric[::SKIP_RESOLUTION]
    ).reset_index(drop=True)
    y_predict_electric_tree_skipped = pd.DataFrame(
        y_predict_electric_tree[::SKIP_RESOLUTION]
    ).reset_index(drop=True)
    x_train_electric_skipped = pd.DataFrame(
        x_train_electric[::SKIP_RESOLUTION]
    ).reset_index(drop=True)
    y_train_electric_skipped = pd.DataFrame(
        y_train_electric[::SKIP_RESOLUTION]
    ).reset_index(drop=True)
    x_test_therm_skipped = pd.DataFrame(x_test_therm[::SKIP_RESOLUTION]).reset_index(
        drop=True
    )
    y_predict_therm_tree_skipped = pd.DataFrame(
        y_predict_therm_tree[::SKIP_RESOLUTION]
    ).reset_index(drop=True)
    x_train_therm_skipped = pd.DataFrame(x_train_therm[::SKIP_RESOLUTION]).reset_index(
        drop=True
    )
    y_train_therm_skipped = pd.DataFrame(y_train_therm[::SKIP_RESOLUTION]).reset_index(
        drop=True
    )

    electric_viz = dtreeviz(
        electric_tree,
        np.asarray(x_test_electric_skipped),
        np.asarray(y_predict_electric_tree_skipped),
        target_name="electric efficiency",
        feature_names=[
            "T_ambient",
            "T_in",
            "mass-flow rate",
            "irradiance",
            "v_wind",
        ],
        X=np.asarray(x_test_electric_skipped.loc[23]),
        orientation="LR",
        precision=5,
        show_just_path=True,
    )
    thermal_viz = dtreeviz(
        thermal_tree,
        np.asarray(x_test_therm_skipped),
        np.asarray(y_predict_therm_tree_skipped),
        target_name="T_out",
        feature_names=[
            "T_ambient",
            "T_in",
            "mass-flow rate",
            "irradiance",
            "v_wind",
        ],
        X=np.asarray(x_test_electric_skipped.loc[23]),
        orientation="LR",
        precision=5,
        show_just_path=True,
    )
    electric_viz.save("electric_decision_tree_test.svg")
    thermal_viz.save("thermal_decision_tree_test.svg")

    electric_viz = dtreeviz(
        electric_tree,
        np.asarray(x_train_electric_skipped),
        np.asarray(y_train_electric_skipped),
        target_name="electric efficiency",
        feature_names=[
            "T_ambient",
            "T_in",
            "mass-flow rate",
            "irradiance",
            "v_wind",
        ],
        X=np.asarray(x_train_electric_skipped.loc[23]),
        orientation="LR",
        precision=5,
        show_just_path=True,
    )
    thermal_viz = dtreeviz(
        thermal_tree,
        np.asarray(x_train_therm_skipped),
        np.asarray(y_train_therm_skipped),
        target_name="T_out",
        feature_names=[
            "T_ambient",
            "T_in",
            "mass-flow rate",
            "irradiance",
            "v_wind",
        ],
        X=np.asarray(x_train_therm_skipped.loc[23]),
        orientation="LR",
        precision=5,
        show_just_path=True,
    )
    electric_viz.save("electric_decision_tree_train.svg")
    thermal_viz.save("thermal_decision_tree_train.svg")

    # Determine the best refressor from the random forest.
    therm_estimator_accuracy: Dict[int, float] = {}
    for index in tqdm(
        range(NUM_ESTIMATORS),
        desc="determining best thermal tree",
        unit="tree",
        leave=True,
    ):
        therm_estimator_accuracy[index] = np.mean(
            np.sqrt(
                (y_test_therm - thermal_forest.estimators_[index].predict(x_test_therm))
                ** 2
            )
        )
    thermal_forest_accuracy = {
        value: key for key, value in therm_estimator_accuracy.items()
    }
    best_thermal_tree = thermal_forest.estimators_[
        thermal_forest_accuracy[min(thermal_forest_accuracy.keys())]
    ]

    electric_forest_accuracy: Dict[int, float] = {}
    for index in tqdm(
        range(NUM_ESTIMATORS),
        desc="determining best electric tree",
        unit="tree",
        leave=True,
    ):
        electric_forest_accuracy[index] = np.mean(
            np.sqrt(
                (
                    y_test_electric
                    - electric_forest.estimators_[index].predict(x_test_electric)
                )
                ** 2
            )
        )
    electric_forest_accuracy = {
        value: key for key, value in electric_forest_accuracy.items()
    }
    best_electric_tree = electric_forest.estimators_[
        electric_forest_accuracy[min(electric_forest_accuracy.keys())]
    ]

    # Predict the values based on these trees and display their accuracies.
    y_predict_best_electric_tree = best_electric_tree.predict(x_test_electric)
    best_electric_error_tree = np.sqrt(
        (y_predict_best_electric_tree - y_test_electric) ** 2
    )
    print(
        f"The best electric tree had a sd of {100 * np.mean(best_electric_error_tree): .3g}% efficiency."
    )
    print(
        f"This compares to a baseline electric sd of {100 * np.mean(electric_error_tree_baseline): .3g}% efficiency."
    )
    best_electric_tree_mape = 100 * (best_electric_error_tree / y_test_electric)
    best_electric_tree_accuracy = 100 - np.mean(best_electric_tree_mape)
    print(
        f"The best electric tree had an accuracy of {best_electric_tree_accuracy: .3g}%."
    )

    y_predict_best_thermal_tree = best_thermal_tree.predict(x_test_therm)
    best_thermal_error_tree = np.sqrt((y_predict_best_thermal_tree - y_test_therm) ** 2)
    print(
        f"The best thermal tree had a sd of {np.mean(best_thermal_error_tree): .3g}degC."
    )
    print(
        f"This compares to a baseline electric sd of {np.mean(thermal_error_tree_baseline): .3g}degC."
    )
    best_thermal_tree_mape = 100 * (best_thermal_error_tree / y_test_therm)
    best_thermal_tree_accuracy = 100 - np.mean(best_thermal_tree_mape)
    print(
        f"The best thermal tree had an accuracy of {best_thermal_tree_accuracy: .3g}%."
    )

    y_predict_best_electric_tree_skipped = best_electric_tree.predict(
        x_test_electric_skipped
    )
    y_predict_best_thermal_tree_skipped = best_thermal_tree.predict(
        x_test_therm_skipped
    )

    # Plot these "best" decision trees.
    electric_viz = dtreeviz(
        best_electric_tree,
        np.asarray(x_test_electric_skipped),
        np.asarray(y_predict_best_electric_tree_skipped),
        target_name="electric efficiency",
        feature_names=[
            "T_ambient",
            "T_in",
            "mass-flow rate",
            "irradiance",
            "v_wind",
        ],
        X=np.asarray(x_test_electric_skipped.loc[23]),
        orientation="LR",
        precision=5,
        show_just_path=True,
    )
    thermal_viz = dtreeviz(
        best_thermal_tree,
        np.asarray(x_test_therm_skipped),
        np.asarray(y_predict_best_thermal_tree_skipped),
        target_name="T_out",
        feature_names=[
            "T_ambient",
            "T_in",
            "mass-flow rate",
            "irradiance",
            "v_wind",
        ],
        X=np.asarray(x_test_electric_skipped.loc[23]),
        orientation="LR",
        precision=5,
        show_just_path=True,
    )
    electric_viz.save("best_electric_decision_tree_test.svg")
    thermal_viz.save("best_thermal_decision_tree_test.svg")

    electric_viz = dtreeviz(
        best_electric_tree,
        np.asarray(x_train_electric_skipped),
        np.asarray(y_train_electric_skipped),
        target_name="electric efficiency",
        feature_names=[
            "T_ambient",
            "T_in",
            "mass-flow rate",
            "irradiance",
            "v_wind",
        ],
        X=np.asarray(x_train_electric_skipped.loc[23]),
        orientation="LR",
        precision=5,
        show_just_path=True,
    )
    thermal_viz = dtreeviz(
        best_thermal_tree,
        np.asarray(x_train_therm_skipped),
        np.asarray(y_train_therm_skipped),
        target_name="T_out",
        feature_names=[
            "T_ambient",
            "T_in",
            "mass-flow rate",
            "irradiance",
            "v_wind",
        ],
        X=np.asarray(x_train_therm_skipped.loc[23]),
        orientation="LR",
        precision=5,
        show_just_path=True,
    )
    electric_viz.save("best_electric_decision_tree_train.svg")
    thermal_viz.save("best_thermal_decision_tree_train.svg")

    with open("best_electric_tree.sav", "wb") as f:
        pickle.dump(best_electric_tree, f)
    with open("best_thermal_tree.sav", "wb") as f:
        pickle.dump(best_thermal_tree, f)


if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])

    # Initial analysis
    # analyse(
    #     parsed_args.data_file_name
    # )

    # Attempt at fitting
    analyse(parsed_args.data_file_name, parsed_args.use_existing_fits)
