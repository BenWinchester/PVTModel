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

from typing import List, Set, Tuple
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.model_selection import cross_val_score, RepeatedKFold

import numpy as np  # type: ignore  # pylint: disable=import-error
import pandas as pd  # type: ignore  # pylint: disable=import-error

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.sparse import data
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
#   Keyword for the electrical efficiency of the collector.
ELECTRICAL_EFFICIENCY: str = "electrical_efficiency"

# Mass-flow rate:
#   Keyword for the mass-flow rate of the collector.
MASS_FLOW_RATE: str = "mass_flow_rate"

# Reconstruction resolution:
#   The resolution to use when reconstructing reduced plots.
RECONSTRUCTION_RESOLUTION: int = 800

# Reduced model:
#   Label to use for reduced model data.
REDUCED_MODEL: str = "reduced model"

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

    return parser.parse_args(args)


def analyse(data_file_name: str) -> None:
    """
    Analysis function for fitting parameters.

    :param data_file_name:
        The data-file name.


    """

    # Parse the input data.
    with open(data_file_name, "r") as f:
        data = json.load(f)

    # Re-structure the data into a Lasso-friendly format.
    if isinstance(data, list):
        processed_data = [
            (
                entry[AMBIENT_TEMPERATURE],
                entry[COLLECTOR_INPUT_TEMPERATURE],
                entry[MASS_FLOW_RATE],
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
    else:
        raise Exception("Input data must be of type `list`. Other formats are not supported.")

    # Reject data where the collector input temperature is greater than zero.
    processed_data = [entry for entry in processed_data if entry[1] < 100]

    # Split the last 50 entries out as test data.
    train_data = processed_data[:-50]
    test_data = processed_data[-50:]

    # Define the variables needed for the fit.
    evaluation_method = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    electrical_model = Lasso(alpha=1)
    thermal_model = Lasso(alpha=1)
    train_data_struct = pd.DataFrame(train_data)
    x_data = train_data_struct[[0, 1, 2, 3, 4]]
    y_elec_data = train_data_struct[6]
    y_therm_data = train_data_struct[5]

    # Train the models on the data.
    print("Fixed alpha at 0.1")
    print("Fitting the models..................... ", end="")
    electrical_model.fit(x_data, y_elec_data)
    thermal_model.fit(x_data, y_therm_data)
    print("[  DONE  ]")

    # Output the model scores.
    print("Generating model scores................ ", end="")
    elec_scores = cross_val_score(electrical_model, x_data, y_elec_data, scoring="neg_mean_absolute_error", cv=evaluation_method, n_jobs=-1)
    therm_scores = cross_val_score(thermal_model, x_data, y_therm_data, scoring="neg_mean_absolute_error", cv=evaluation_method, n_jobs=-1)
    print("[  DONE  ]")
    print(f"Electrical model scores: {np.mean(np.abs(elec_scores)):.3g} ({np.std(np.abs(elec_scores)):.3g})")
    print(f"Thermal model scores: {np.mean(np.abs(therm_scores)):.3g} ({np.std(np.abs(therm_scores)):.3g})")

    # Attempt to fit the data with a tuned alpha value.
    variable_alpha_electrical_model = LassoCV(alphas=np.arange(1e-9, 1e-7, 1e-9), cv=evaluation_method, n_jobs=-1)
    variable_alpha_thermal_model = LassoCV(alphas=np.arange(3e-7, 4e-7, 1e-9), cv=evaluation_method, n_jobs=-1)

    # Train the models on the data.
    print("Varying alpha values")
    print("Fitting the models..................... ", end="")
    variable_alpha_electrical_model.fit(x_data, y_elec_data)
    variable_alpha_thermal_model.fit(x_data, y_therm_data)
    print("[  DONE  ]")

    print(f"Electrical alpha value: {variable_alpha_electrical_model.alpha_:.3g}")
    print(f"Thermal alpha value: {variable_alpha_thermal_model.alpha_:.3g}")

    # Output the model scores.
    print("Re-running at suggested alpha values... ", end="")
    electrical_model = Lasso(alpha=variable_alpha_electrical_model.alpha_)
    thermal_model = Lasso(alpha=variable_alpha_thermal_model.alpha_)
    electrical_model.fit(x_data, y_elec_data)
    thermal_model.fit(x_data, y_therm_data)
    print("[  DONE  ]")
    print("Generating model scores................ ", end="")
    elec_scores = cross_val_score(electrical_model, x_data, y_elec_data, scoring="neg_mean_absolute_error", cv=evaluation_method, n_jobs=-1)
    therm_scores = cross_val_score(thermal_model, x_data, y_therm_data, scoring="neg_mean_absolute_error", cv=evaluation_method, n_jobs=-1)
    print("[  DONE  ]")
    print(f"Electrical model scores: {np.mean(np.abs(elec_scores)):.3g} ({np.std(np.abs(elec_scores)):.3g})")
    print(f"Thermal model scores: {np.mean(np.abs(therm_scores)):.3g} ({np.std(np.abs(therm_scores)):.3g})")

    print(f"Electrical alpha value: {variable_alpha_electrical_model.alpha_:.3g}")
    print(f"Thermal alpha value: {variable_alpha_thermal_model.alpha_:.3g}")

    # Predict sone new data based on the test data.
    test_data_struct = pd.DataFrame(test_data)
    test_x_data = test_data_struct[[0, 1, 2, 3, 4]]
    test_technical_electrical = test_data_struct[6]    
    test_technical_thermal = test_data_struct[5]    
    predicted_electrical = electrical_model.predict(test_x_data)
    predicted_thermal = thermal_model.predict(test_x_data)

    # Plot the predicted and generated data.
    plt.scatter(test_x_data[0], test_technical_electrical, label="technical", marker="x")
    plt.scatter(test_x_data[0], predicted_electrical, label="reduced", marker="x")
    plt.xlabel("Data point")
    plt.ylabel("Electrical efficiency")
    plt.title("Selection of points chosen for model comparison")
    plt.legend()
    plt.savefig("electrical_efficiency_ai_fitting.png", transparent=True)
    plt.close()

    plt.scatter(test_x_data[0], test_technical_thermal, label="technical", marker="x")
    plt.scatter(test_x_data[0], predicted_thermal, label="reduced", marker="x")
    plt.xlabel("Data point")
    plt.ylabel("Collector output temperature")
    plt.title("Selection of points chosen for model comparison")
    plt.legend()
    plt.savefig("thermal_efficiency_ai_fitting.png", transparent=True)
    plt.close()
    
    # print("Re-running at alpha=1................ ", end="")
    # electrical_model = Ridge(alpha=1e-9)
    # thermal_model = Ridge(alpha=3.09e-7)
    # electrical_model.fit(x_data, y_elec_data)
    # thermal_model.fit(x_data, y_therm_data)
    # print("[  DONE  ]")
    # print("Generating model scores................ ", end="")
    # elec_scores = cross_val_score(electrical_model, x_data, y_elec_data, scoring="neg_mean_absolute_error", cv=evaluation_method, n_jobs=-1)
    # therm_scores = cross_val_score(thermal_model, x_data, y_therm_data, scoring="neg_mean_absolute_error", cv=evaluation_method, n_jobs=-1)
    # print("[  DONE  ]")

    print(f"Electrical model scores: {np.mean(np.abs(elec_scores)):.3g} ({np.std(np.abs(elec_scores)):.3g})")
    print(f"Thermal model scores: {np.mean(np.abs(therm_scores)):.3g} ({np.std(np.abs(therm_scores)):.3g})")

    # Save the models.
    with open("electrical_model.sav", "wb") as f:
        pickle.dump(electrical_model, f)
    with open("thermal_model.sav", "wb") as f:
        pickle.dump(thermal_model, f)

if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])

    # Initial analysis
    # analyse(
    #     parsed_args.data_file_name
    # )

    # Attempt at fitting
    analyse(parsed_args.data_file_name)
