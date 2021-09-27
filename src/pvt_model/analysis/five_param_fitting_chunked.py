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
import json
import re
import sys

from typing import List, Set, Tuple

import numpy as np  # type: ignore  # pylint: disable=import-error

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

    # Determine the various sets.
    ambient_temperatures: Set[float] = {entry[AMBIENT_TEMPERATURE] for entry in data}
    collector_input_temperatures: Set[float] = {
        entry[COLLECTOR_INPUT_TEMPERATURE] for entry in data
    }
    mass_flow_rates: Set[float] = {entry[MASS_FLOW_RATE] for entry in data}
    solar_irradiances: Set[float] = {entry[SOLAR_IRRADIANCE] for entry in data}
    wind_speeds: Set[float] = {entry[WIND_SPEED] for entry in data}

    ###############
    # WIND SPEEDS #
    ###############

    # Set up a variable for holding the polyfits.
    polyfits = []

    for T_amb in tqdm(
        ambient_temperatures, desc="ambient_temperature", leave=False, unit="value"
    ):
        for T_c_in in collector_input_temperatures:
            for m_dot in mass_flow_rates:
                for G in solar_irradiances:
                    reduced_runs = {
                        entry[WIND_SPEED]: entry[THERMAL_EFFICIENCY]
                        for entry in data
                        if entry[AMBIENT_TEMPERATURE] == T_amb
                        and entry[COLLECTOR_INPUT_TEMPERATURE] == T_c_in
                        and entry[MASS_FLOW_RATE] == m_dot
                        and entry[SOLAR_IRRADIANCE] == G
                        and entry[THERMAL_EFFICIENCY] is not None
                    }
                    if len(reduced_runs) <= 1:
                        continue
                    fit = np.polyfit(
                        list(reduced_runs.keys()), list(reduced_runs.values()), 2
                    )
                    polyfits.append(fit)

    fig, axs = plt.subplots(2, 2)
    ax = axs[0, 0]
    ax.ticklabel_format(useOffset=False)
    first = [entry[0] for entry in polyfits]
    ax.hist(first, bins=50)
    ax.set_title("First coefficient")

    ax = axs[0, 1]
    ax.ticklabel_format(useOffset=False)
    second = [entry[1] for entry in polyfits]
    ax.hist(second, bins=50)
    ax.set_title("Second coefficient")

    ax = axs[1, 0]
    ax.ticklabel_format(useOffset=False)
    third = [entry[2] for entry in polyfits]
    ax.hist(third, bins=50)
    ax.set_title("Third coefficient")

    # ax = axs[1, 1]
    # ax.ticklabel_format(useOffset=False)
    # fourth = [entry[3] for entry in polyfits]
    # ax.hist(fourth, bins=50)
    # ax.set_title("Fourth coefficient")

    # print(f"Fit of wind speed gives {np.mean(first):.3f}v_w^3 + {np.mean(second):.3f}v_w^2 + {np.mean(third):.3f}v_2 + {np.mean(fourth):.3f}")
    print(
        f"Fit of wind speed gives {np.mean(first):.3f}v_w^2 + {np.mean(second):.3f}v_w + {np.mean(third):.3f}"
    )

    fig.suptitle("Wind speed fitting")
    plt.show()

    ###################
    # MASS FLOW RATES #
    ###################

    # Set up a variable for holding the polyfits.
    polyfits = []

    for T_amb in tqdm(
        ambient_temperatures, desc="ambient_temperature", leave=False, unit="value"
    ):
        for T_c_in in collector_input_temperatures:
            for G in solar_irradiances:
                for v_w in wind_speeds:
                    reduced_runs = {
                        entry[MASS_FLOW_RATE]: entry[THERMAL_EFFICIENCY]
                        for entry in data
                        if entry[AMBIENT_TEMPERATURE] == T_amb
                        and entry[COLLECTOR_INPUT_TEMPERATURE] == T_c_in
                        and entry[SOLAR_IRRADIANCE] == G
                        and entry[WIND_SPEED] == v_w
                        and entry[THERMAL_EFFICIENCY] is not None
                    }
                    if len(reduced_runs) <= 1:
                        continue
                    fit = np.polyfit(
                        list(reduced_runs.keys()), list(reduced_runs.values()), 2
                    )
                    polyfits.append(fit)

    fig, axs = plt.subplots(2, 2)
    ax = axs[0, 0]
    ax.ticklabel_format(useOffset=False)
    first = [entry[0] for entry in polyfits]
    ax.hist(first, bins=50)
    ax.set_title("First coefficient")

    ax = axs[0, 1]
    ax.ticklabel_format(useOffset=False)
    second = [entry[1] for entry in polyfits]
    ax.hist(second, bins=50)
    ax.set_title("Second coefficient")

    ax = axs[1, 0]
    ax.ticklabel_format(useOffset=False)
    third = [entry[2] for entry in polyfits]
    ax.hist(third, bins=50)
    ax.set_title("Third coefficient")

    # ax = axs[1, 1]
    # ax.ticklabel_format(useOffset=False)
    # fourth = [entry[3] for entry in polyfits]
    # ax.hist(fourth, bins=50)
    # ax.set_title("Fourth coefficient")

    # print(f"Fit of mass flow rates gives {np.mean(first):.3f}\\dot(m)^3 + {np.mean(second):.3f}\\dot(m)^2 + {np.mean(third):.3f}\\dot(m) + {np.mean(fourth)}")
    print(
        f"Fit of mass flow rates gives {np.mean(first):.3f}\\dot(m)^2 + {np.mean(second):.3f}\\dot(m) + {np.mean(third):.3f}"
    )

    fig.suptitle("Mass flow-rate fitting")
    plt.show()

    ######################
    # INPUT TEMPERATURES #
    ######################

    # Set up a variable for holding the polyfits.
    polyfits = []

    for T_amb in tqdm(
        ambient_temperatures, desc="ambient_temperature", leave=False, unit="value"
    ):
        for m_dot in mass_flow_rates:
            for G in solar_irradiances:
                for v_w in wind_speeds:
                    reduced_runs = {
                        entry[COLLECTOR_INPUT_TEMPERATURE]: entry[THERMAL_EFFICIENCY]
                        for entry in data
                        if entry[AMBIENT_TEMPERATURE] == T_amb
                        and entry[MASS_FLOW_RATE] == m_dot
                        and entry[SOLAR_IRRADIANCE] == G
                        and entry[WIND_SPEED] == v_w
                        and entry[THERMAL_EFFICIENCY] is not None
                    }
                    if len(reduced_runs) <= 1:
                        continue
                    fit = np.polyfit(
                        list(reduced_runs.keys()), list(reduced_runs.values()), 2
                    )
                    polyfits.append(fit)

    fig, axs = plt.subplots(2, 2)
    ax = axs[0, 0]
    ax.ticklabel_format(useOffset=False)
    first = [entry[0] for entry in polyfits]
    ax.hist(first, bins=50)
    ax.set_title("First coefficient")

    ax = axs[0, 0]
    ax.ticklabel_format(useOffset=False)
    second = [entry[1] for entry in polyfits]
    ax.hist(second, bins=50)
    ax.set_title("Second coefficient")

    ax = axs[0, 0]
    ax.ticklabel_format(useOffset=False)
    third = [entry[2] for entry in polyfits]
    ax.hist(third, bins=50)
    ax.set_title("Third coefficient")

    # ax = axs[0, 0]
    # ax.ticklabel_format(useOffset=False)
    # fourth = [entry[3] for entry in polyfits]
    # ax.hist(fourth, bins=50)
    # ax.set_title("Fourth coefficient")

    # print(f"Fit of collector input temperatures gives {np.mean(first):.3f}T_c_in^3 + {np.mean(second):.3f}T_c_in^2 + {np.mean(third):.3f}T_C_in + {np.mean(fourth):.3f}")
    print(
        f"Fit of collector input temperatures gives {np.mean(first):.3f}T_c_in^2 + {np.mean(second):.3f}T_c_in + {np.mean(third):.3f}"
    )

    fig.suptitle("Input temperature fitting")
    plt.show()


def _best_guess(
    input_runs: Tuple[List[float], List[float], List[float], List[float], List[float]],
    a_0: float,
    a_1: float,
    a_2: float,
    a_3: float,
    a_4: float,
    a_5: float,
    a_6: float,
    a_7: float,
    a_8: float,
    a_9: float,
    a_10: float,
    a_11: float,
    a_12: float,
    a_13: float,
    a_14: float,
    a_15: float,
    a_16: float,
    a_17: float,
    a_18: float,
    a_19: float,
    a_20: float,
    a_21: float,
    a_22: float,
    a_23: float,
    a_24: float,
    a_25: float,
    a_26: float,
    a_27: float,
    a_28: float,
    a_29: float,
    a_30: float,
    a_31: float,
    a_32: float,
    a_33: float,
    a_34: float,
    a_35: float,
    a_36: float,
    a_37: float,
    a_38: float,
    a_39: float,
    a_40: float,
    a_41: float,
    a_42: float,
    a_43: float,
    a_44: float,
) -> List[float]:
    """
    Attempts a best-guess solution

    :param input_runs:
        The input run information.

    :params: a_1, a_2, ...
        Parameters used for specifying the fit.

    """

    (
        ambient_temperature,
        collector_input_temperature,
        mass_flow_rate,
        solar_irradiance,
        wind_speed,
    ) = input_runs

    return (
        a_0
        + a_1 * np.log(solar_irradiance)
        + a_2 * (np.log(solar_irradiance)) ** 2
        + a_3 * np.log(mass_flow_rate)
        + a_4 * (np.log(mass_flow_rate)) ** 2
        + a_5 * np.log(solar_irradiance) * np.log(mass_flow_rate)
        + a_6 * wind_speed
        + a_7 * wind_speed ** 2
        + a_8 * wind_speed ** 3
        + a_9 * wind_speed ** 4
        + a_10 * wind_speed ** 5
        + a_11 * wind_speed * np.log(mass_flow_rate)
        + a_12 * wind_speed * np.log(solar_irradiance)
        + a_13 * wind_speed ** 2 * np.log(mass_flow_rate)
        + a_14 * wind_speed ** 2 * np.log(solar_irradiance)
        + ambient_temperature
        * (
            a_15
            + a_16 * np.log(solar_irradiance)
            + a_17 * (np.log(solar_irradiance)) ** 2
            + a_18 * np.log(mass_flow_rate)
            + a_19 * (np.log(mass_flow_rate)) ** 2
            + a_20 * np.log(solar_irradiance) * np.log(mass_flow_rate)
            + a_21 * wind_speed
            + a_22 * wind_speed ** 2
            + a_23 * wind_speed ** 3
            + a_24 * wind_speed ** 4
            + a_25 * wind_speed ** 5
            + a_26 * wind_speed * np.log(mass_flow_rate)
            + a_27 * wind_speed * np.log(solar_irradiance)
            + a_28 * wind_speed ** 2 * np.log(mass_flow_rate)
            + a_29 * wind_speed ** 2 * np.log(solar_irradiance)
        )
        + collector_input_temperature
        * (
            a_30
            + a_31 * np.log(solar_irradiance)
            + a_32 * (np.log(solar_irradiance)) ** 2
            + a_33 * np.log(mass_flow_rate)
            + a_34 * (np.log(mass_flow_rate)) ** 2
            + a_35 * np.log(solar_irradiance) * np.log(mass_flow_rate)
            + a_36 * wind_speed
            + a_37 * wind_speed ** 2
            + a_38 * wind_speed ** 3
            + a_39 * wind_speed ** 4
            + a_40 * wind_speed ** 5
            + a_41 * wind_speed * np.log(mass_flow_rate)
            + a_42 * wind_speed * np.log(solar_irradiance)
            + a_43 * wind_speed ** 2 * np.log(mass_flow_rate)
            + a_44 * wind_speed ** 2 * np.log(solar_irradiance)
        )
    )


def _plot(
    ambient_temperatures: List[float],
    collector_input_temperatures: List[float],
    data_type: str,
    mass_flow_rates: List[float],
    results: List[np.ndarray],
    solar_irradiances: List[float],
    y_data: List[float],
    wind_speeds: List[float],
) -> None:
    """
    Plots the various outputs.

    :param data_type:
        The data type being plotted to display on the y axes.

    :param results:
        The results of the curve fitting.

    """

    print(
        f"Fitted curve params:\n- first chunk: {results[0][0]}\n- second chunk: {results[1][0]}\n- third chunk: {results[2][0]}"
    )
    print(
        "Fitted curve for first chunk: {a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
            a=results[0][0][0],
            b=results[0][0][1],
            c=results[0][0][2],
        )
        + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
            a=results[0][0][3],
            b=results[0][0][4],
            c=results[0][0][5],
        )
        + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
            a=results[0][0][6],
            b=results[0][0][7],
            c=results[0][0][8],
        )
        + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
            a=results[0][0][9],
            b=results[0][0][10],
        )
        + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[0][0][11],
            b=results[0][0][12],
        )
        + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[0][0][13],
            b=results[0][0][14],
        )
        + "+ T_amb * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
            a=results[0][0][15],
            b=results[0][0][16],
            c=results[0][0][17],
        )
        + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
            a=results[0][0][18],
            b=results[0][0][19],
            c=results[0][0][20],
        )
        + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
            a=results[0][0][21],
            b=results[0][0][22],
            c=results[0][0][23],
        )
        + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
            a=results[0][0][24],
            b=results[0][0][25],
        )
        + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[0][0][26],
            b=results[0][0][27],
        )
        + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) )) ".format(
            a=results[0][0][28],
            b=results[0][0][29],
        )
        + "+ T_c,in * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
            a=results[0][0][30],
            b=results[0][0][31],
            c=results[0][0][32],
        )
        + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
            a=results[0][0][33],
            b=results[0][0][34],
            c=results[0][0][35],
        )
        + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
            a=results[0][0][36],
            b=results[0][0][37],
            c=results[0][0][38],
        )
        + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
            a=results[0][0][39],
            b=results[0][0][40],
        )
        + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[0][0][41],
            b=results[0][0][42],
        )
        + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) )) ".format(
            a=results[0][0][43],
            b=results[0][0][44],
        )
    )
    print(
        "Fitted curve for second chunk: {a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
            a=results[1][0][0],
            b=results[1][0][1],
            c=results[1][0][2],
        )
        + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
            a=results[1][0][3],
            b=results[1][0][4],
            c=results[1][0][5],
        )
        + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
            a=results[1][0][6],
            b=results[1][0][7],
            c=results[1][0][8],
        )
        + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
            a=results[1][0][9],
            b=results[1][0][10],
        )
        + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[1][0][11],
            b=results[1][0][12],
        )
        + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[1][0][13],
            b=results[1][0][14],
        )
        + "+ T_amb * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
            a=results[1][0][15],
            b=results[1][0][16],
            c=results[1][0][17],
        )
        + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
            a=results[1][0][18],
            b=results[1][0][19],
            c=results[1][0][20],
        )
        + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
            a=results[1][0][21],
            b=results[1][0][22],
            c=results[1][0][23],
        )
        + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
            a=results[1][0][24],
            b=results[1][0][25],
        )
        + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[1][0][26],
            b=results[1][0][27],
        )
        + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) )) ".format(
            a=results[1][0][28],
            b=results[1][0][29],
        )
        + "+ T_c,in * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
            a=results[1][0][30],
            b=results[1][0][31],
            c=results[1][0][32],
        )
        + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
            a=results[1][0][33],
            b=results[1][0][34],
            c=results[1][0][35],
        )
        + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
            a=results[1][0][36],
            b=results[1][0][37],
            c=results[1][0][38],
        )
        + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
            a=results[1][0][39],
            b=results[1][0][40],
        )
        + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[1][0][41],
            b=results[1][0][42],
        )
        + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) )) ".format(
            a=results[1][0][43],
            b=results[1][0][44],
        )
    )
    print(
        "Fitted curve for third chunk: {a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
            a=results[2][0][0],
            b=results[2][0][1],
            c=results[2][0][2],
        )
        + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
            a=results[2][0][3],
            b=results[2][0][4],
            c=results[2][0][5],
        )
        + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
            a=results[2][0][6],
            b=results[2][0][7],
            c=results[2][0][8],
        )
        + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
            a=results[2][0][9],
            b=results[2][0][10],
        )
        + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[2][0][11],
            b=results[2][0][12],
        )
        + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[2][0][13],
            b=results[2][0][14],
        )
        + "+ T_amb * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
            a=results[2][0][15],
            b=results[2][0][16],
            c=results[2][0][17],
        )
        + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
            a=results[2][0][18],
            b=results[2][0][19],
            c=results[2][0][20],
        )
        + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
            a=results[2][0][21],
            b=results[2][0][22],
            c=results[2][0][23],
        )
        + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
            a=results[2][0][24],
            b=results[2][0][25],
        )
        + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[2][0][26],
            b=results[2][0][27],
        )
        + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) )) ".format(
            a=results[2][0][28],
            b=results[2][0][29],
        )
        + "+ T_c,in * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
            a=results[2][0][30],
            b=results[2][0][31],
            c=results[2][0][32],
        )
        + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
            a=results[2][0][33],
            b=results[2][0][34],
            c=results[2][0][35],
        )
        + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
            a=results[2][0][36],
            b=results[2][0][37],
            c=results[2][0][38],
        )
        + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
            a=results[2][0][39],
            b=results[2][0][40],
        )
        + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
            a=results[2][0][41],
            b=results[2][0][42],
        )
        + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) )) ".format(
            a=results[2][0][43],
            b=results[2][0][44],
        )
    )

    # Compute the chunk-by-chunk best-guess data.
    best_guess_data = []
    for index, collector_input_temperature in enumerate(collector_input_temperatures):
        if collector_input_temperature < 10:
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[0][0],
                )
            )
            continue
        if collector_input_temperature < 20:
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[1][0],
                )
            )
            continue
        if collector_input_temperature < 30:
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[2][0],
                )
            )
            continue
        if collector_input_temperature < 40:
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[3][0],
                )
            )
            continue
        if collector_input_temperature < 60:
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[4][0],
                )
            )
            continue
        if collector_input_temperature < 70:
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[5][0],
                )
            )
            continue
        if collector_input_temperature < 80:
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[6][0],
                )
            )
            continue
        if collector_input_temperature < 100:
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[7][0],
                )
            )

    plt.scatter(ambient_temperatures, y_data, label=TECHNICAL_MODEL)
    plt.scatter(
        ambient_temperatures,
        best_guess_data,
        label=REDUCED_MODEL,
    )
    plt.xlabel("Ambient temeprature / degC")
    plt.ylabel(data_type)
    plt.legend()
    plt.title("Ambient temperature reconstruction")
    plt.show()

    plt.scatter(collector_input_temperatures, y_data, label=TECHNICAL_MODEL)
    plt.scatter(
        collector_input_temperatures,
        best_guess_data,
        label=REDUCED_MODEL,
    )
    plt.legend()
    plt.xlabel("Collector input temperature / degC")
    plt.ylabel(data_type)
    plt.title("Collector input temperature reconstruction")
    plt.show()

    plt.scatter(mass_flow_rates, y_data, label=TECHNICAL_MODEL)
    plt.scatter(
        mass_flow_rates,
        best_guess_data,
        label=REDUCED_MODEL,
    )
    plt.legend()
    plt.xlabel("Mass flow rate / litres/hour")
    plt.ylabel(data_type)
    plt.title("Mass-flow rate reconstruction")
    plt.show()

    plt.scatter(solar_irradiances, y_data, label=TECHNICAL_MODEL)
    plt.scatter(
        solar_irradiances,
        best_guess_data,
        label=REDUCED_MODEL,
    )
    plt.legend()
    plt.xlabel("Solar irradiance / W/m^2")
    plt.ylabel(data_type)
    plt.title("Solar irradiance reconstruction")
    plt.show()

    plt.scatter(wind_speeds, y_data, label=TECHNICAL_MODEL)
    plt.scatter(
        wind_speeds,
        best_guess_data,
        label=REDUCED_MODEL,
    )
    plt.xlabel("Wind speed / m/2")
    plt.ylabel(data_type)
    plt.legend()
    plt.title("Wind speed reconstruction")
    plt.show()

    plt.scatter(
        ambient_temperatures[::RECONSTRUCTION_RESOLUTION],
        y_data[::RECONSTRUCTION_RESOLUTION],
        label=TECHNICAL_MODEL,
        marker="x",
    )
    plt.scatter(
        ambient_temperatures[::RECONSTRUCTION_RESOLUTION],
        best_guess_data[::RECONSTRUCTION_RESOLUTION],
        label=REDUCED_MODEL,
        marker="x",
    )
    plt.xlabel("Ambient temeprature / degC")
    plt.ylabel(data_type)
    plt.legend()
    plt.title("Select ambient temperature reconstruction points")
    plt.show()

    plt.scatter(
        collector_input_temperatures[::RECONSTRUCTION_RESOLUTION],
        y_data[::RECONSTRUCTION_RESOLUTION],
        label=TECHNICAL_MODEL,
        marker="x",
    )
    plt.scatter(
        collector_input_temperatures[::RECONSTRUCTION_RESOLUTION],
        best_guess_data[::RECONSTRUCTION_RESOLUTION],
        label=REDUCED_MODEL,
        marker="x",
    )
    plt.xlabel("Collector input temeprature / degC")
    plt.ylabel(data_type)
    plt.legend()
    plt.title("Select collector input temperature reconstruction points")
    plt.show()

    plt.scatter(
        mass_flow_rates[::RECONSTRUCTION_RESOLUTION],
        y_data[::RECONSTRUCTION_RESOLUTION],
        label=TECHNICAL_MODEL,
        marker="x",
    )
    plt.scatter(
        mass_flow_rates[::RECONSTRUCTION_RESOLUTION],
        best_guess_data[::RECONSTRUCTION_RESOLUTION],
        label=REDUCED_MODEL,
        marker="x",
    )
    plt.xlabel("Mass flow rate / litres/hour")
    plt.ylabel(data_type)
    plt.legend()
    plt.title("Select mass-flow rate reconstruction points")
    plt.show()

    plt.scatter(
        solar_irradiances[::RECONSTRUCTION_RESOLUTION],
        y_data[::RECONSTRUCTION_RESOLUTION],
        label=TECHNICAL_MODEL,
        marker="x",
    )
    plt.scatter(
        solar_irradiances[::RECONSTRUCTION_RESOLUTION],
        best_guess_data[::RECONSTRUCTION_RESOLUTION],
        label=REDUCED_MODEL,
        marker="x",
    )
    plt.xlabel("Solar irradiance / W/m^2")
    plt.ylabel(data_type)
    plt.legend()
    plt.title("Select solar irradiance reconstruction points")
    plt.show()

    plt.scatter(
        wind_speeds[::RECONSTRUCTION_RESOLUTION],
        y_data[::RECONSTRUCTION_RESOLUTION],
        label=TECHNICAL_MODEL,
        marker="x",
    )
    plt.scatter(
        wind_speeds[::RECONSTRUCTION_RESOLUTION],
        best_guess_data[::RECONSTRUCTION_RESOLUTION],
        label=REDUCED_MODEL,
        marker="x",
    )
    plt.xlabel("Wind speed / m/s")
    plt.ylabel(data_type)
    plt.legend()
    plt.title("Select wind speed reconstruction points")
    plt.show()


def fit(data_file_name: str) -> None:
    """
    Attempts to generate a fit for the various parameters involved.

    :param data_file_name:
        The name of the input data file to parse.

    """

    # Parse the input data.
    with open(data_file_name, "r") as f:
        data = json.load(f)

    # Transform the data to tuples.
    processed_data = [
        (
            entry[AMBIENT_TEMPERATURE],
            entry[COLLECTOR_INPUT_TEMPERATURE],
            entry[ELECTRICAL_EFFICIENCY],
            entry[MASS_FLOW_RATE],
            entry[SOLAR_IRRADIANCE],
            entry[THERMAL_EFFICIENCY],
            entry[WIND_SPEED],
        )
        for entry in data
        if entry[AMBIENT_TEMPERATURE] is not None
        and entry[COLLECTOR_INPUT_TEMPERATURE] is not None
        and entry[ELECTRICAL_EFFICIENCY] is not None
        and entry[MASS_FLOW_RATE] is not None
        and entry[SOLAR_IRRADIANCE] is not None
        and entry[THERMAL_EFFICIENCY] is not None
        and entry[WIND_SPEED] is not None
        and entry[COLLECTOR_INPUT_TEMPERATURE] <= 100
    ]

    ambient_temperatures = [entry[0] for entry in processed_data]
    collector_input_temperatures = [entry[1] for entry in processed_data]
    electrical_efficiencies = [entry[2] for entry in processed_data]
    mass_flow_rates = [3600 * entry[3] for entry in processed_data]
    solar_irradiances = [entry[4] for entry in processed_data]
    thermal_efficiencies = [entry[5] for entry in processed_data]
    wind_speeds = [entry[6] for entry in processed_data]

    ambient_temperatures_first_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if collector_input_temperatures[index] < 10
    ]
    collector_input_temperatures_first_chunk = [
        entry for entry in collector_input_temperatures if entry < 10
    ]
    mass_flow_rates_first_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if collector_input_temperatures[index] < 10
    ]
    solar_irradiances_first_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if collector_input_temperatures[index] < 10
    ]
    wind_speeds_first_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if collector_input_temperatures[index] < 10
    ]

    ambient_temperatures_second_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 10 <= collector_input_temperatures[index] < 20
    ]
    collector_input_temperatures_second_chunk = [
        entry for entry in collector_input_temperatures if 10 <= entry < 20
    ]
    mass_flow_rates_second_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 10 <= collector_input_temperatures[index] < 20
    ]
    solar_irradiances_second_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 10 <= collector_input_temperatures[index] < 20
    ]
    wind_speeds_second_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 10 <= collector_input_temperatures[index] < 20
    ]

    ambient_temperatures_third_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 20 <= collector_input_temperatures[index] < 30
    ]
    collector_input_temperatures_third_chunk = [
        entry for entry in collector_input_temperatures if 20 <= entry < 30
    ]
    mass_flow_rates_third_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 20 <= collector_input_temperatures[index] < 30
    ]
    solar_irradiances_third_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 20 <= collector_input_temperatures[index] < 30
    ]
    wind_speeds_third_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 20 <= collector_input_temperatures[index] < 30
    ]

    ambient_temperatures_fourth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 30 <= collector_input_temperatures[index] < 40
    ]
    collector_input_temperatures_fourth_chunk = [
        entry for entry in collector_input_temperatures if 30 <= entry < 40
    ]
    mass_flow_rates_fourth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 30 <= collector_input_temperatures[index] < 40
    ]
    solar_irradiances_fourth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 30 <= collector_input_temperatures[index] < 40
    ]
    wind_speeds_fourth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 30 <= collector_input_temperatures[index] < 40
    ]

    ambient_temperatures_fifth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 40 <= collector_input_temperatures[index] < 60
    ]
    collector_input_temperatures_fifth_chunk = [
        entry for entry in collector_input_temperatures if 40 <= entry < 60
    ]
    mass_flow_rates_fifth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 40 <= collector_input_temperatures[index] < 60
    ]
    solar_irradiances_fifth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 40 <= collector_input_temperatures[index] < 60
    ]
    wind_speeds_fifth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 40 <= collector_input_temperatures[index] < 60
    ]

    ambient_temperatures_sixth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 60 <= collector_input_temperatures[index] < 70
    ]
    collector_input_temperatures_sixth_chunk = [
        entry for entry in collector_input_temperatures if 60 <= entry < 70
    ]
    mass_flow_rates_sixth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 60 <= collector_input_temperatures[index] < 70
    ]
    solar_irradiances_sixth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 60 <= collector_input_temperatures[index] < 70
    ]
    wind_speeds_sixth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 60 <= collector_input_temperatures[index] < 70
    ]

    ambient_temperatures_seventh_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 70 <= collector_input_temperatures[index] < 80
    ]
    collector_input_temperatures_seventh_chunk = [
        entry for entry in collector_input_temperatures if 70 <= entry < 80
    ]
    mass_flow_rates_seventh_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 70 <= collector_input_temperatures[index] < 80
    ]
    solar_irradiances_seventh_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 70 <= collector_input_temperatures[index] < 80
    ]
    wind_speeds_seventh_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 70 <= collector_input_temperatures[index] < 80
    ]

    ambient_temperatures_eigth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 80 <= collector_input_temperatures[index] <= 100
    ]
    collector_input_temperatures_eigth_chunk = [
        entry for entry in collector_input_temperatures if 80 <= entry <= 100
    ]
    mass_flow_rates_eigth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 80 <= collector_input_temperatures[index] <= 100
    ]
    solar_irradiances_eigth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 80 <= collector_input_temperatures[index] <= 100
    ]
    wind_speeds_eigth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 80 <= collector_input_temperatures[index] <= 100
    ]

    # Set up initial guesses for the parameters.
    initial_guesses = (
        -2.8,
        1,
        -0.05,
        0.4,
        -0.1,
        0.005,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    print("Computing fit for first chunk ........... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    first_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_first_chunk,
            collector_input_temperatures_first_chunk,
            mass_flow_rates_first_chunk,
            solar_irradiances_first_chunk,
            wind_speeds_first_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if collector_input_temperatures[index] < 10
        ],
        initial_guesses,
    )

    first_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_first_chunk,
            collector_input_temperatures_first_chunk,
            mass_flow_rates_first_chunk,
            solar_irradiances_first_chunk,
            wind_speeds_first_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if collector_input_temperatures[index] < 10
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for second chunk .......... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    second_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_second_chunk,
            collector_input_temperatures_second_chunk,
            mass_flow_rates_second_chunk,
            solar_irradiances_second_chunk,
            wind_speeds_second_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 10 <= collector_input_temperatures[index] < 20
        ],
        initial_guesses,
    )

    second_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_second_chunk,
            collector_input_temperatures_second_chunk,
            mass_flow_rates_second_chunk,
            solar_irradiances_second_chunk,
            wind_speeds_second_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 10 <= collector_input_temperatures[index] < 20
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for third chunk ........... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    third_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_third_chunk,
            collector_input_temperatures_third_chunk,
            mass_flow_rates_third_chunk,
            solar_irradiances_third_chunk,
            wind_speeds_third_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 20 <= collector_input_temperatures[index] < 30
        ],
        initial_guesses,
    )

    third_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_third_chunk,
            collector_input_temperatures_third_chunk,
            mass_flow_rates_third_chunk,
            solar_irradiances_third_chunk,
            wind_speeds_third_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 20 <= collector_input_temperatures[index] < 30
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for fourth chunk .......... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    fourth_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_fourth_chunk,
            collector_input_temperatures_fourth_chunk,
            mass_flow_rates_fourth_chunk,
            solar_irradiances_fourth_chunk,
            wind_speeds_fourth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 30 <= collector_input_temperatures[index] < 40
        ],
        initial_guesses,
    )

    fourth_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_fourth_chunk,
            collector_input_temperatures_fourth_chunk,
            mass_flow_rates_fourth_chunk,
            solar_irradiances_fourth_chunk,
            wind_speeds_fourth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 30 <= collector_input_temperatures[index] < 40
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for fifth chunk ........... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    fifth_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_fifth_chunk,
            collector_input_temperatures_fifth_chunk,
            mass_flow_rates_fifth_chunk,
            solar_irradiances_fifth_chunk,
            wind_speeds_fifth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 40 <= collector_input_temperatures[index] < 60
        ],
        initial_guesses,
    )

    fifth_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_fifth_chunk,
            collector_input_temperatures_fifth_chunk,
            mass_flow_rates_fifth_chunk,
            solar_irradiances_fifth_chunk,
            wind_speeds_fifth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 40 <= collector_input_temperatures[index] < 60
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for sixth chunk ........... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    sixth_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_sixth_chunk,
            collector_input_temperatures_sixth_chunk,
            mass_flow_rates_sixth_chunk,
            solar_irradiances_sixth_chunk,
            wind_speeds_sixth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 60 <= collector_input_temperatures[index] < 70
        ],
        initial_guesses,
    )

    sixth_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_sixth_chunk,
            collector_input_temperatures_sixth_chunk,
            mass_flow_rates_sixth_chunk,
            solar_irradiances_sixth_chunk,
            wind_speeds_sixth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 60 <= collector_input_temperatures[index] < 70
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for seventh chunk ......... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    seventh_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_seventh_chunk,
            collector_input_temperatures_seventh_chunk,
            mass_flow_rates_seventh_chunk,
            solar_irradiances_seventh_chunk,
            wind_speeds_seventh_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 70 <= collector_input_temperatures[index] < 80
        ],
        initial_guesses,
    )

    seventh_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_seventh_chunk,
            collector_input_temperatures_seventh_chunk,
            mass_flow_rates_seventh_chunk,
            solar_irradiances_seventh_chunk,
            wind_speeds_seventh_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 70 <= collector_input_temperatures[index] < 80
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for eigth chunk ........... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    eigth_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_eigth_chunk,
            collector_input_temperatures_eigth_chunk,
            mass_flow_rates_eigth_chunk,
            solar_irradiances_eigth_chunk,
            wind_speeds_eigth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 80 <= collector_input_temperatures[index] <= 100
        ],
        initial_guesses,
    )

    eigth_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_eigth_chunk,
            collector_input_temperatures_eigth_chunk,
            mass_flow_rates_eigth_chunk,
            solar_irradiances_eigth_chunk,
            wind_speeds_eigth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 80 <= collector_input_temperatures[index] <= 100
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    # Plot the various outputs.
    _plot(
        ambient_temperatures,
        collector_input_temperatures,
        "Thermal efficiency",
        mass_flow_rates,
        [
            first_thermal_efficiency_results,
            second_thermal_efficiency_results,
            third_thermal_efficiency_results,
            fourth_thermal_efficiency_results,
            fifth_thermal_efficiency_results,
            sixth_thermal_efficiency_results,
            seventh_thermal_efficiency_results,
            eigth_thermal_efficiency_results,
        ],
        solar_irradiances,
        thermal_efficiencies,
        wind_speeds,
    )

    _plot(
        ambient_temperatures,
        collector_input_temperatures,
        "Electrical efficiency",
        mass_flow_rates,
        [
            first_electrical_efficiency_results,
            second_electrical_efficiency_results,
            third_electrical_efficiency_results,
            fourth_electrical_efficiency_results,
            fifth_electrical_efficiency_results,
            sixth_electrical_efficiency_results,
            seventh_electrical_efficiency_results,
            eigth_electrical_efficiency_results,
        ],
        solar_irradiances,
        electrical_efficiencies,
        wind_speeds,
    )


if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])

    # Initial analysis
    # analyse(
    #     parsed_args.data_file_name
    # )

    # Attempt at fitting
    fit(parsed_args.data_file_name)
