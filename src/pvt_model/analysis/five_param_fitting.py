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
    # a_26: float,
    # a_27: float,
    # a_28: float,
    # a_29: float,
    # a_30: float,
    # a_31: float,
    # a_32: float,
    # a_33: float,
    # a_34: float,
    # a_35: float,
    # a_36: float,
    # a_37: float,
    # a_38: float,
    # a_39: float,
    # a_40: float,
    # a_41: float,
    # a_42: float,
    # a_43: float,
    # a_44: float,
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
        + (
            (a_1 * np.log(solar_irradiance) + a_2 * (np.log(solar_irradiance)) ** 2)
            + (
                a_3 * np.log(mass_flow_rate)
                # a_3 * mass_flow_rate
                + a_4 * (np.log(mass_flow_rate)) ** 2
                # + a_4 * (1 - np.exp(-a_4 ** 2 / mass_flow_rate)) ** 2
                # + a_4 * mass_flow_rate * (1 - np.exp(-a_4 ** 2 / mass_flow_rate)) ** 2
            )
            + a_5 * np.log(mass_flow_rate) * np.log(solar_irradiance)
        )
        + ambient_temperature
        * (
            (a_6 * np.log(solar_irradiance) + a_7 * (np.log(solar_irradiance)) ** 2)
            + (
                a_8 * np.log(mass_flow_rate)
                # a_12 * mass_flow_rate
                + a_9 * (np.log(mass_flow_rate)) ** 2
                # + a_13 * (1 - np.exp(-a_13 ** 2 / mass_flow_rate)) ** 2
                # + a_13 * mass_flow_rate * (1 - np.exp(-a_13 ** 2 / mass_flow_rate)) ** 2
            )
            + a_10 * np.log(solar_irradiance) * np.log(mass_flow_rate)
        )
        + wind_speed ** 0.16
        * (
            (a_11 * np.log(solar_irradiance) + a_12 * (np.log(solar_irradiance)) ** 2)
            + (
                a_13 * np.log(mass_flow_rate)
                # a_12 * mass_flow_rate
                + a_14 * (np.log(mass_flow_rate)) ** 2
                # + a_13 * (1 - np.exp(-a_13 ** 2 / mass_flow_rate)) ** 2
                # + a_13 * mass_flow_rate * (1 - np.exp(-a_13 ** 2 / mass_flow_rate)) ** 2
            )
            + a_15 * np.log(solar_irradiance) * np.log(mass_flow_rate)
        )
        + ambient_temperature
        * wind_speed ** 0.16
        * (
            (a_16 * np.log(solar_irradiance) + a_17 * (np.log(solar_irradiance)) ** 2)
            + (
                a_18 * np.log(mass_flow_rate)
                # a_12 * mass_flow_rate
                + a_19 * (np.log(mass_flow_rate)) ** 2
                # + a_13 * (1 - np.exp(-a_13 ** 2 / mass_flow_rate)) ** 2
                # + a_13 * mass_flow_rate * (1 - np.exp(-a_13 ** 2 / mass_flow_rate)) ** 2
            )
            + a_20 * np.log(solar_irradiance) * np.log(mass_flow_rate)
        )
        + collector_input_temperature
        * (
            (a_21 * np.log(solar_irradiance) + a_22 * (np.log(solar_irradiance)) ** 2)
            + (
                a_23 * np.log(mass_flow_rate)
                # a_21 * mass_flow_rate
                + a_24 * (np.log(mass_flow_rate)) ** 2
                # + a_22 * (1 - np.exp(-a_22 ** 2 / mass_flow_rate)) ** 2
                # + a_22 * mass_flow_rate * (1 - np.exp(-a_22 ** 2 / mass_flow_rate)) ** 2
            )
            + a_25 * np.log(solar_irradiance) * np.log(mass_flow_rate)
        )
    )

    # return (
    #     a_0
    #     + a_1 * np.log(solar_irradiance)
    #     + a_2 * (np.log(solar_irradiance)) ** 2
    #     + a_3 * np.log(mass_flow_rate)
    #     + a_4 * (np.log(mass_flow_rate)) ** 2
    #     + a_5 * np.log(solar_irradiance) * np.log(mass_flow_rate)
    #     + a_6 * wind_speed
    #     + a_7 * wind_speed ** 2
    #     + a_8 * wind_speed ** 3
    #     + a_9 * wind_speed ** 4
    #     + a_10 * wind_speed ** 5
    #     + a_11 * wind_speed * np.log(mass_flow_rate)
    #     + a_12 * wind_speed * np.log(solar_irradiance)
    #     + a_13 * wind_speed ** 2 * np.log(mass_flow_rate)
    #     + a_14 * wind_speed ** 2 * np.log(solar_irradiance)
    #     + ambient_temperature
    #     * (
    #         a_15
    #         + a_16 * np.log(solar_irradiance)
    #         + a_17 * (np.log(solar_irradiance)) ** 2
    #         + a_18 * np.log(mass_flow_rate)
    #         + a_19 * (np.log(mass_flow_rate)) ** 2
    #         + a_20 * np.log(solar_irradiance) * np.log(mass_flow_rate)
    #         + a_21 * wind_speed
    #         + a_22 * wind_speed ** 2
    #         + a_23 * wind_speed ** 3
    #         + a_24 * wind_speed ** 4
    #         + a_25 * wind_speed ** 5
    #         + a_26 * wind_speed * np.log(mass_flow_rate)
    #         + a_27 * wind_speed * np.log(solar_irradiance)
    #         + a_28 * wind_speed ** 2 * np.log(mass_flow_rate)
    #         + a_29 * wind_speed ** 2 * np.log(solar_irradiance)
    #     )
    #     + collector_input_temperature
    #     * (
    #         a_30
    #         + a_31 * np.log(solar_irradiance)
    #         + a_32 * (np.log(solar_irradiance)) ** 2
    #         + a_33 * np.log(mass_flow_rate)
    #         + a_34 * (np.log(mass_flow_rate)) ** 2
    #         + a_35 * np.log(solar_irradiance) * np.log(mass_flow_rate)
    #         + a_36 * wind_speed
    #         + a_37 * wind_speed ** 2
    #         + a_38 * wind_speed ** 3
    #         + a_39 * wind_speed ** 4
    #         + a_40 * wind_speed ** 5
    #         + a_41 * wind_speed * np.log(mass_flow_rate)
    #         + a_42 * wind_speed * np.log(solar_irradiance)
    #         + a_43 * wind_speed ** 2 * np.log(mass_flow_rate)
    #         + a_44 * wind_speed ** 2 * np.log(solar_irradiance)
    #     )
    # )


def _plot(
    ambient_temperatures: List[float],
    collector_input_temperatures: List[float],
    data_type: str,
    mass_flow_rates: List[float],
    results: np.ndarray,
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

    # print(
    #     "Fitted curve: {a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
    #         a=results[0][0],
    #         b=results[0][1],
    #         c=results[0][2],
    #     )
    #     + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
    #         a=results[0][3],
    #         b=results[0][4],
    #         c=results[0][5],
    #     )
    #     + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
    #         a=results[0][6],
    #         b=results[0][7],
    #         c=results[0][8],
    #     )
    #     + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
    #         a=results[0][9],
    #         b=results[0][10],
    #     )
    #     + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
    #         a=results[0][11],
    #         b=results[0][12],
    #     )
    #     + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
    #         a=results[0][13],
    #         b=results[0][14],
    #     )
    #     + "+ T_amb * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
    #         a=results[0][15],
    #         b=results[0][16],
    #         c=results[0][17],
    #     )
    #     + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
    #         a=results[0][18],
    #         b=results[0][19],
    #         c=results[0][20],
    #     )
    #     + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
    #         a=results[0][21],
    #         b=results[0][22],
    #         c=results[0][23],
    #     )
    #     + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
    #         a=results[0][24],
    #         b=results[0][25],
    #     )
    #     + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
    #         a=results[0][26],
    #         b=results[0][27],
    #     )
    #     + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) )) ".format(
    #         a=results[0][28],
    #         b=results[0][29],
    #     )
    #     + "+ T_c,in * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
    #         a=results[0][30],
    #         b=results[0][31],
    #         c=results[0][32],
    #     )
    #     + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
    #         a=results[0][33],
    #         b=results[0][34],
    #         c=results[0][35],
    #     )
    #     + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 ".format(
    #         a=results[0][36],
    #         b=results[0][37],
    #         c=results[0][38],
    #     )
    #     + "+ {a:.2g}v_w^4 + {b:.2g}v_w^5 ".format(
    #         a=results[0][39],
    #         b=results[0][40],
    #     )
    #     + "+ v_w * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) ) ".format(
    #         a=results[0][41],
    #         b=results[0][42],
    #     )
    #     + "+ v_w^2 * ({a:.2g}ln(m_dot) + {b:.2g}ln(G) )) ".format(
    #         a=results[0][43],
    #         b=results[0][44],
    #     )
    # )

    params = [
        "-> fit #1:\n{params}".format(
            params="\n".join(
                [
                    "{sub_index}:  -> {value}: {sd}".format(
                        value=results[0][sub_index],
                        sd=np.sqrt(np.diag(np.ma.masked_invalid(results[1])))[
                            sub_index
                        ],
                        sub_index=sub_index,
                    )
                    for sub_index in range(len(results[0]))
                ]
            ),
        )
    ]

    filename = re.sub("-| |:", "_", str(datetime.datetime.now()))

    with open(f"fitted_parameters_{filename}.txt", "w") as f:
        f.write("\n -> ".join(params))
    print("Fitted curve params:\n{}".format("\n - ".join(params)))

    plt.scatter(ambient_temperatures, y_data, label=TECHNICAL_MODEL)
    plt.scatter(
        ambient_temperatures,
        _best_guess(
            (
                np.array(ambient_temperatures),
                np.array(collector_input_temperatures),
                np.array(mass_flow_rates),
                np.array(solar_irradiances),
                np.array(wind_speeds),
            ),
            *results[0],
        ),
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
        _best_guess(
            (
                np.array(ambient_temperatures),
                np.array(collector_input_temperatures),
                np.array(mass_flow_rates),
                np.array(solar_irradiances),
                np.array(wind_speeds),
            ),
            *results[0],
        ),
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
        _best_guess(
            (
                np.array(ambient_temperatures),
                np.array(collector_input_temperatures),
                np.array(mass_flow_rates),
                np.array(solar_irradiances),
                np.array(wind_speeds),
            ),
            *results[0],
        ),
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
        _best_guess(
            (
                np.array(ambient_temperatures),
                np.array(collector_input_temperatures),
                np.array(mass_flow_rates),
                np.array(solar_irradiances),
                np.array(wind_speeds),
            ),
            *results[0],
        ),
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
        _best_guess(
            (
                np.array(ambient_temperatures),
                np.array(collector_input_temperatures),
                np.array(mass_flow_rates),
                np.array(solar_irradiances),
                np.array(wind_speeds),
            ),
            *results[0],
        ),
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
        _best_guess(
            (
                np.array(ambient_temperatures[::RECONSTRUCTION_RESOLUTION]),
                np.array(collector_input_temperatures)[::RECONSTRUCTION_RESOLUTION],
                np.array(mass_flow_rates[::RECONSTRUCTION_RESOLUTION]),
                np.array(solar_irradiances[::RECONSTRUCTION_RESOLUTION]),
                np.array(wind_speeds[::RECONSTRUCTION_RESOLUTION]),
            ),
            *results[0],
        ),
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
        _best_guess(
            (
                np.array(ambient_temperatures[::RECONSTRUCTION_RESOLUTION]),
                np.array(collector_input_temperatures)[::RECONSTRUCTION_RESOLUTION],
                np.array(mass_flow_rates[::RECONSTRUCTION_RESOLUTION]),
                np.array(solar_irradiances[::RECONSTRUCTION_RESOLUTION]),
                np.array(wind_speeds[::RECONSTRUCTION_RESOLUTION]),
            ),
            *results[0],
        ),
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
        _best_guess(
            (
                np.array(ambient_temperatures[::RECONSTRUCTION_RESOLUTION]),
                np.array(collector_input_temperatures)[::RECONSTRUCTION_RESOLUTION],
                np.array(mass_flow_rates[::RECONSTRUCTION_RESOLUTION]),
                np.array(solar_irradiances[::RECONSTRUCTION_RESOLUTION]),
                np.array(wind_speeds[::RECONSTRUCTION_RESOLUTION]),
            ),
            *results[0],
        ),
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
        _best_guess(
            (
                np.array(ambient_temperatures[::RECONSTRUCTION_RESOLUTION]),
                np.array(collector_input_temperatures)[::RECONSTRUCTION_RESOLUTION],
                np.array(mass_flow_rates[::RECONSTRUCTION_RESOLUTION]),
                np.array(solar_irradiances[::RECONSTRUCTION_RESOLUTION]),
                np.array(wind_speeds[::RECONSTRUCTION_RESOLUTION]),
            ),
            *results[0],
        ),
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
        _best_guess(
            (
                np.array(ambient_temperatures[::RECONSTRUCTION_RESOLUTION]),
                np.array(collector_input_temperatures)[::RECONSTRUCTION_RESOLUTION],
                np.array(mass_flow_rates[::RECONSTRUCTION_RESOLUTION]),
                np.array(solar_irradiances[::RECONSTRUCTION_RESOLUTION]),
                np.array(wind_speeds[::RECONSTRUCTION_RESOLUTION]),
            ),
            *results[0],
        ),
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
    ]

    ambient_temperatures = [entry[0] for entry in processed_data]
    collector_input_temperatures = [entry[1] for entry in processed_data]
    electrical_efficiencies = [entry[2] for entry in processed_data]
    mass_flow_rates = [3600 * entry[3] for entry in processed_data]
    solar_irradiances = [entry[4] for entry in processed_data]
    thermal_efficiencies = [entry[5] for entry in processed_data]
    wind_speeds = [entry[6] for entry in processed_data]

    # Set up initial guesses for the parameters.
    initial_guesses = (
        0.5,  # a_0
        1,  # a_1
        -0.05,  # a_2
        0.4,  # a_3
        -0.1,  # a_4
        0.6215,  # a_5
        -11.37,  # a_6
        0.3965,  # a_7
        0,  # a_8
        0,  # a_9
        0,  # a_10
        0,  # a_11
        0,  # a_12
        0,  # a_13
        0,  # a_14
        0,  # a_15
        0,  # a_16
        0,  # a_17
        0,  # a_18
        0,  # a_19
        0,  # a_20
        0,  # a_21
        0,  # a_22
        0,  # a_23
        0,  # a_24
        0,  # a_25
        # 0,  # a_26
        # 0,  # a_27
        # 0,  # a_28
        # 0,  # a_29
        # 0,  # a_30
        # 0,  # a_31
        # 0,  # a_32
        # # 0,  # a_33
        # 0,  # a_34
        # 0,  # a_35
        # 0,  # a_36
        # 0,  # a_37
        # 0,  # a_38
        # 0,  # a_39
        # 0,  # a_40
        # 0,  # a_41
        # 0,  # a_42
        # 0,  # a_43
        # 0,  # a_44
    )

    # Attempt a curve fit.
    thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures,
            collector_input_temperatures,
            mass_flow_rates,
            solar_irradiances,
            wind_speeds,
        ),
        thermal_efficiencies,
        initial_guesses,
    )

    electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures,
            collector_input_temperatures,
            mass_flow_rates,
            solar_irradiances,
            wind_speeds,
        ),
        electrical_efficiencies,
        initial_guesses,
    )

    # Plot the various outputs.
    _plot(
        ambient_temperatures,
        collector_input_temperatures,
        "Thermal efficiency",
        mass_flow_rates,
        thermal_efficiency_results,
        solar_irradiances,
        thermal_efficiencies,
        wind_speeds,
    )

    _plot(
        ambient_temperatures,
        collector_input_temperatures,
        "Electrical efficiency",
        mass_flow_rates,
        electrical_efficiency_results,
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
