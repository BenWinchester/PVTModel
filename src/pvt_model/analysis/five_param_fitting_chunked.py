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
    # a_19: float,
    # a_20: float,
    # a_21: float,
    a_22: float,
    a_23: float,
    a_24: float,
    a_25: float,
    a_26: float,
    a_27: float,
    a_28: float,
    a_29: float,
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
        + a_1 * np.log(solar_irradiance)
        + a_2 * (np.log(solar_irradiance)) ** 2
        + a_3 * mass_flow_rate * (1 - np.exp(-a_3 / mass_flow_rate))
        + a_4 * np.log(solar_irradiance) * mass_flow_rate * (1 - np.exp(-a_5 / mass_flow_rate))
        + a_6 * (wind_speed ** 3 + a_7) ** (1 / 3)
        + a_8 * wind_speed * (
            + a_9 * mass_flow_rate * (1 - np.exp(-a_9 / mass_flow_rate))
            + a_10 * np.log(solar_irradiance)
        )
        + ambient_temperature
        * (
            a_11
            + a_12 * np.log(solar_irradiance)
            + a_13 * (np.log(solar_irradiance)) ** 2
            + a_14 * mass_flow_rate * (1 - np.exp(-a_14 / mass_flow_rate))
            + a_15 * np.log(solar_irradiance) * mass_flow_rate * (1 - np.exp(-a_16 / mass_flow_rate))
            + a_17 * (wind_speed ** 3 + a_18) ** (1 / 3)
            # + a_19 * wind_speed * (
            #     + a_20 * mass_flow_rate * (1 - np.exp(-a_20 / mass_flow_rate))
            #     + a_21 * np.log(solar_irradiance)
            # )
        )
        + collector_input_temperature
        * (
            a_22
            + a_23 * np.log(solar_irradiance)
            + a_24 * (np.log(solar_irradiance)) ** 2
            + a_25 * mass_flow_rate * (1 - np.exp(-a_25 / mass_flow_rate))
            + a_26 * np.log(solar_irradiance) * mass_flow_rate * (1 - np.exp(-a_27 / mass_flow_rate))
            + a_28 * (wind_speed ** 3 + a_29) ** (1 / 3)
        #     + a_30 * wind_speed * (
        #         + a_31 * mass_flow_rate * (1 - np.exp(-a_31 / mass_flow_rate))
        #         + a_32 * np.log(solar_irradiance)
        #     )
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

    # Plot the various outputs.
    print("Saving output ........................... ", end="")

    import pdb

    pdb.set_trace()

    params = [
        "- fit #{fit_number}:\n{params}".format(
            fit_number=index,
            params="\n".join(
                [
                    "  - {value}: {sd}".format(
                        value=entry[0][sub_index],
                        sd=np.sqrt(
                            np.diag(np.ma.masked_invalid(entry[1]))
                        )[sub_index],
                    )
                    for sub_index in range(48)
                ]
            ),
        )
        for index, entry in enumerate(results)
    ]

    filename = re.sub("-| |:", "_", str(datetime.datetime.now()))

    with open(f"fitted_parameters_{filename}.txt", "w") as f:
        f.write("\n - ".join(params))

    print("[  DONE  ]")

    print(
        f"Fitted curve params:\n- first chunk: {results[0][0]}\n- second chunk: {results[1][0]}\n- third chunk: {results[2][0]}"
    )
    for index, chunk_name in enumerate(
        [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eigth",
            "ninth",
            "tenth",
            "eleventh",
            "twelth",
        ]
    ):
        print(
            "Fitted curve for {name} chunk: {a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
                name=chunk_name,
                a=results[index][0][0],
                b=results[index][0][1],
                c=results[index][0][2],
            )
            + "+ {a:.2g}m_dot + {b:.2g}m_dot^2 + {c:.2g}m_dot * ln(G) ".format(
                a=results[index][0][3],
                b=results[index][0][4],
                c=results[index][0][5],
            )
            + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 + {d:.2g}v_2^4 ".format(
                a=results[index][0][6],
                b=results[index][0][7],
                c=results[index][0][8],
                d=results[index][0][9],
            )
            + "+ v_w * ({a:.2g}m_dot + {b:.2g}m_dot^2 + {c:.2g}ln(G) ) ".format(
                a=results[index][0][10],
                b=results[index][0][11],
                c=results[index][0][12],
            )
            + "+ v_w^2 * ({a:.2g}m_dot + {b:.2g}m_dot^2 + {b:.2g}ln(G) ) ".format(
                a=results[index][0][13],
                b=results[index][0][45],
                c=results[index][0][14],
            )
            + "+ T_amb * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
                a=results[index][0][15],
                b=results[index][0][16],
                c=results[index][0][17],
            )
            + "+ {a:.2g}ln(m_dot) + {b:.2g}|ln(m_dot)|^2 + {c:.2g}ln(m_dot) * ln(G) ".format(
                a=results[index][0][18],
                b=results[index][0][19],
                c=results[index][0][20],
            )
            + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 + {d:.2g}v_w^4 ".format(
                a=results[index][0][21],
                b=results[index][0][22],
                c=results[index][0][23],
                d=results[index][0][24],
            )
            + "+ v_w * ({a:.2g}m_dot + {b:.2g}m_dot^2 + {c:.2g}ln(G) ) ".format(
                a=results[index][0][25],
                b=results[index][0][26],
                c=results[index][0][27],
            )
            + "+ v_w^2 * ({a:.2g}m_dot + {b:.2g}m_dot^2 + {c:.2g}ln(G) )) ".format(
                a=results[index][0][28],
                b=results[index][0][46],
                c=results[index][0][29],
            )
            + "+ T_c,in * ({a:.2g} + {b:.2g}ln(G) + {c:.2g}|ln(G)|^2 ".format(
                a=results[index][0][30],
                b=results[index][0][31],
                c=results[index][0][32],
            )
            + "+ {a:.2g}m_dot + {b:.2g}m_dot^2 + {c:.2g}m_dot * ln(G) ".format(
                a=results[index][0][33],
                b=results[index][0][34],
                c=results[index][0][35],
            )
            + "+ {a:.2g}v_w + {b:.2g}v_w^2 + {c:.2g}v_w^3 + {d:.2g}v_w^4 ".format(
                a=results[index][0][36],
                b=results[index][0][37],
                c=results[index][0][38],
                d=results[index][0][39],
            )
            + "+ v_w * ({a:.2g}m_dot + {b:.2g}m_dot^2 + {c:.2g}ln(G) ) ".format(
                a=results[index][0][40],
                b=results[index][0][41],
                c=results[index][0][42],
            )
            + "+ v_w^2 * ({a:.2g}m_dot + {b:.2g}m_dot^2 + {c:.2g}ln(G) )) ".format(
                a=results[index][0][43],
                b=results[index][0][47],
                c=results[index][0][44],
            )
        )

    print("Computing best-guess data ............... ", end="")
    # Compute the chunk-by-chunk best-guess data.
    best_guess_data = []
    for index, collector_input_temperature in enumerate(collector_input_temperatures):
        if ambient_temperatures[index] < 15 and collector_input_temperature < 20:
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
        if 15 <= ambient_temperatures[index] < 30 and collector_input_temperature < 20:
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
        if 30 <= ambient_temperatures[index] <= 45 and collector_input_temperature < 20:
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
        if ambient_temperatures[index] < 15 and 20 <= collector_input_temperature < 40:
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
        if (
            15 <= ambient_temperatures[index] < 30
            and 20 <= collector_input_temperature < 40
        ):
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
        if (
            30 <= ambient_temperatures[index] <= 45
            and 20 <= collector_input_temperature < 40
        ):
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
        if ambient_temperatures[index] < 15 and 40 <= collector_input_temperature < 60:
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
        if (
            15 <= ambient_temperatures[index] < 30
            and 40 <= collector_input_temperature < 60
        ):
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
        if (
            30 <= ambient_temperatures[index] <= 45
            and 40 <= collector_input_temperature < 60
        ):
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[8][0],
                )
            )
        if ambient_temperatures[index] < 15 and 60 <= collector_input_temperature < 80:
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[9][0],
                )
            )
        if (
            15 <= ambient_temperatures[index] < 30
            and 60 <= collector_input_temperature < 80
        ):
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[10][0],
                )
            )
        if (
            30 <= ambient_temperatures[index] <= 45
            and 60 <= collector_input_temperature < 80
        ):
            best_guess_data.append(
                _best_guess(
                    (
                        np.array(ambient_temperatures[index]),
                        np.array(collector_input_temperature),
                        np.array(mass_flow_rates[index]),
                        np.array(solar_irradiances[index]),
                        np.array(wind_speeds[index]),
                    ),
                    *results[11][0],
                )
            )

    print("[  DONE  ]")

    # Compute a histogram of the accuracy of the data.
    print("Computing percentage accuracies ......... ", end="")
    percentage_accuracies = (
        100 * (np.array(best_guess_data) - np.array(y_data)) / np.array(y_data)
    )
    mean = np.mean(percentage_accuracies)
    variance = np.sum((percentage_accuracies - mean) ** 2) / len(percentage_accuracies)
    print("[  DONE  ]")
    print(f"Reduced model {data_type} accuracy, mean={mean:.2g}, var={variance:.2g}")
    print(f"Max percentage accuracy: {max(percentage_accuracies):.2g}")
    print(f"Min percentage accuracy: {min(percentage_accuracies):.2g}")

    # Reduce the range to be limited by +/- 100
    percentage_accuracies = [
        entry for entry in percentage_accuracies if -100 < entry < 100
    ]
    print(f"Max capped percentage accuracy: {max(percentage_accuracies):.2g}")
    print(f"Min capped percentage accuracy: {min(percentage_accuracies):.2g}")

    mean = np.mean(percentage_accuracies)
    variance = np.sum((percentage_accuracies - mean) ** 2) / len(percentage_accuracies)

    # Plot the histogram of the accuracy of the data
    print("Computing accuracy histogram ............ ", end="")
    ax = plt.gca()
    plt.hist(percentage_accuracies, bins=list(range(-100, 101, 1)))
    plt.xlabel("Percentage accuracy")
    plt.ylabel("Frequency")
    ax.set_yscale("log")
    plt.title(
        f"Capped reduced model {data_type} accuracy, mean={mean:.2g}, var={variance:.2g}"
    )
    plt.show()

    print("[  DONE  ]")
    print(
        f"Capped reduced model {data_type} accuracy, mean={mean:.2g}, var={variance:.2g}"
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


def _partial_fit(
    ambient_temperatures: List[float],
    collector_input_temperatures: List[float],
    electrical_efficiencies: List[float],
    identifier: str,
    mass_flow_rates: List[float],
    solar_irradiances: List[float],
    thermal_efficiencies: List[float],
    wind_speeds: List[float],
) -> None:
    """
    Compute a partial fit using some of the data.

    """

    print(f"Carrying out partial fit: {identifier}")

    # Chunks are as follows:
    # 1: T_amb:   [0, 15)
    #    T_c,in:  [0, 20)
    # 2: T_amb:   [15, 30)
    #    T_c,in:  [0, 20)
    # 3: T_amb:   [30, 45)
    #    T_c,in:  [0, 20)
    # 4: T_amb:   [0, 15)
    #    T_c,in:  [20, 40)
    # 5: T_amb:   [15, 30)
    #    T_c,in:  [20, 40)
    # 6: T_amb:   [30, 45)
    #    T_c,in:  [20, 40)
    # 7: T_amb:   [0, 15)
    #    T_c,in:  [40, 60)
    # 8: T_amb:   [15, 30)
    #    T_c,in:  [40, 60)
    # 9: T_amb:   [30, 45)
    #    T_c,in:  [40, 60)
    # 10: T_amb:  [0, 15)
    #     T_c,in: [60, 80)
    # 11: T_amb:  [15, 30)
    #     T_c,in: [60, 80)
    # 12: T_amb:  [30, 45)
    #     T_c,in: [60, 80)

    ambient_temperatures_first_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if entry < 15 and collector_input_temperatures[index] < 20
    ]
    collector_input_temperatures_first_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if ambient_temperatures[index] < 15 and entry < 20
    ]
    mass_flow_rates_first_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if ambient_temperatures[index] < 15 and collector_input_temperatures[index] < 20
    ]
    solar_irradiances_first_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if ambient_temperatures[index] < 15 and collector_input_temperatures[index] < 20
    ]
    wind_speeds_first_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if ambient_temperatures[index] < 15 and collector_input_temperatures[index] < 20
    ]

    ambient_temperatures_second_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 15 <= entry < 30 and collector_input_temperatures[index] < 20
    ]
    collector_input_temperatures_second_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if 15 <= ambient_temperatures[index] < 30 and entry < 20
    ]
    mass_flow_rates_second_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 15 <= ambient_temperatures[index] < 30
        and collector_input_temperatures[index] < 20
    ]
    solar_irradiances_second_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 15 <= ambient_temperatures[index] < 30
        and collector_input_temperatures[index] < 20
    ]
    wind_speeds_second_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 15 <= ambient_temperatures[index] < 30
        and collector_input_temperatures[index] < 20
    ]

    ambient_temperatures_third_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 30 <= entry <= 45 and collector_input_temperatures[index] < 20
    ]
    collector_input_temperatures_third_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if 30 <= ambient_temperatures[index] <= 45 and entry < 20
    ]
    mass_flow_rates_third_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 30 <= ambient_temperatures[index] <= 45
        and collector_input_temperatures[index] < 20
    ]
    solar_irradiances_third_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 30 <= ambient_temperatures[index] <= 45
        and collector_input_temperatures[index] < 20
    ]
    wind_speeds_third_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 30 <= ambient_temperatures[index] <= 45
        and collector_input_temperatures[index] < 20
    ]

    ambient_temperatures_fourth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if entry < 15 and 20 <= collector_input_temperatures[index] < 40
    ]
    collector_input_temperatures_fourth_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if ambient_temperatures[index] < 15 and 20 <= entry < 40
    ]
    mass_flow_rates_fourth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if ambient_temperatures[index] < 15
        and 20 <= collector_input_temperatures[index] < 40
    ]
    solar_irradiances_fourth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if ambient_temperatures[index] < 15
        and 20 <= collector_input_temperatures[index] < 40
    ]
    wind_speeds_fourth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if ambient_temperatures[index] < 15
        and 20 <= collector_input_temperatures[index] < 40
    ]

    ambient_temperatures_fifth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 15 <= entry < 30 and 20 <= collector_input_temperatures[index] < 40
    ]
    collector_input_temperatures_fifth_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if 15 <= ambient_temperatures[index] < 30 and 20 <= entry < 40
    ]
    mass_flow_rates_fifth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 15 <= ambient_temperatures[index] < 30
        and 20 <= collector_input_temperatures[index] < 40
    ]
    solar_irradiances_fifth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 15 <= ambient_temperatures[index] < 30
        and 20 <= collector_input_temperatures[index] < 40
    ]
    wind_speeds_fifth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 15 <= ambient_temperatures[index] < 30
        and 20 <= collector_input_temperatures[index] < 40
    ]

    ambient_temperatures_sixth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 30 < entry <= 45 and 20 <= collector_input_temperatures[index] < 40
    ]
    collector_input_temperatures_sixth_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if 30 <= ambient_temperatures[index] <= 45 and 20 <= entry < 40
    ]
    mass_flow_rates_sixth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 30 <= ambient_temperatures[index] <= 45
        and 20 <= collector_input_temperatures[index] < 40
    ]
    solar_irradiances_sixth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 30 <= ambient_temperatures[index] <= 45
        and 20 <= collector_input_temperatures[index] < 40
    ]
    wind_speeds_sixth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 30 <= ambient_temperatures[index] <= 45
        and 20 <= collector_input_temperatures[index] < 40
    ]

    ambient_temperatures_seventh_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if entry < 15 and 40 <= collector_input_temperatures[index] < 60
    ]
    collector_input_temperatures_seventh_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if ambient_temperatures[index] < 15 and 40 <= entry < 60
    ]
    mass_flow_rates_seventh_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if ambient_temperatures[index] < 15
        and 40 <= collector_input_temperatures[index] < 60
    ]
    solar_irradiances_seventh_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if ambient_temperatures[index] < 15
        and 40 <= collector_input_temperatures[index] < 60
    ]
    wind_speeds_seventh_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if ambient_temperatures[index] < 15
        and 40 <= collector_input_temperatures[index] < 60
    ]

    ambient_temperatures_eigth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 15 <= entry < 30 and 40 <= collector_input_temperatures[index] < 60
    ]
    collector_input_temperatures_eigth_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if 15 <= ambient_temperatures[index] < 30 and 40 <= entry < 60
    ]
    mass_flow_rates_eigth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 15 <= ambient_temperatures[index] < 30
        and 40 <= collector_input_temperatures[index] < 60
    ]
    solar_irradiances_eigth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 15 <= ambient_temperatures[index] < 30
        and 40 <= collector_input_temperatures[index] < 60
    ]
    wind_speeds_eigth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 15 <= ambient_temperatures[index] < 30
        and 40 <= collector_input_temperatures[index] < 60
    ]

    ambient_temperatures_ninth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 30 <= entry <= 45 and 40 <= collector_input_temperatures[index] < 60
    ]
    collector_input_temperatures_ninth_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if 30 <= ambient_temperatures[index] <= 45 and 40 <= entry < 60
    ]
    mass_flow_rates_ninth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 30 <= ambient_temperatures[index] <= 45
        and 40 <= collector_input_temperatures[index] < 60
    ]
    solar_irradiances_ninth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 30 <= ambient_temperatures[index] <= 45
        and 40 <= collector_input_temperatures[index] < 60
    ]
    wind_speeds_ninth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 30 <= ambient_temperatures[index] <= 45
        and 40 <= collector_input_temperatures[index] < 60
    ]

    ambient_temperatures_tenth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if entry < 15 and 60 <= collector_input_temperatures[index] < 80
    ]
    collector_input_temperatures_tenth_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if ambient_temperatures[index] < 15 and 60 <= entry < 80
    ]
    mass_flow_rates_tenth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if ambient_temperatures[index] < 15
        and 60 <= collector_input_temperatures[index] < 80
    ]
    solar_irradiances_tenth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if ambient_temperatures[index] < 15
        and 60 <= collector_input_temperatures[index] < 80
    ]
    wind_speeds_tenth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if ambient_temperatures[index] < 15
        and 60 <= collector_input_temperatures[index] < 80
    ]

    ambient_temperatures_eleventh_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 15 <= entry < 30 and 60 <= collector_input_temperatures[index] < 80
    ]
    collector_input_temperatures_eleventh_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if 15 <= ambient_temperatures[index] < 30 and 60 <= entry < 80
    ]
    mass_flow_rates_eleventh_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 15 <= ambient_temperatures[index] < 30
        and 60 <= collector_input_temperatures[index] < 80
    ]
    solar_irradiances_eleventh_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 15 <= ambient_temperatures[index] < 30
        and 60 <= collector_input_temperatures[index] < 80
    ]
    wind_speeds_eleventh_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 15 <= ambient_temperatures[index] < 30
        and 60 <= collector_input_temperatures[index] < 80
    ]

    ambient_temperatures_twelth_chunk = [
        entry
        for index, entry in enumerate(ambient_temperatures)
        if 30 <= entry <= 45 and 60 <= collector_input_temperatures[index] < 80
    ]
    collector_input_temperatures_twelth_chunk = [
        entry
        for index, entry in enumerate(collector_input_temperatures)
        if 30 <= ambient_temperatures[index] <= 45 and 60 <= entry < 80
    ]
    mass_flow_rates_twelth_chunk = [
        entry
        for index, entry in enumerate(mass_flow_rates)
        if 30 <= ambient_temperatures[index] <= 45
        and 60 <= collector_input_temperatures[index] < 80
    ]
    solar_irradiances_twelth_chunk = [
        entry
        for index, entry in enumerate(solar_irradiances)
        if 30 <= ambient_temperatures[index] <= 45
        and 60 <= collector_input_temperatures[index] < 80
    ]
    wind_speeds_twelth_chunk = [
        entry
        for index, entry in enumerate(wind_speeds)
        if 30 <= ambient_temperatures[index] <= 45
        and 60 <= collector_input_temperatures[index] < 80
    ]

    # Set up initial guesses for the parameters.
    initial_guesses = (
        0, # a_0
        0, # a_1
        0, # a_2
        0, # a_3
        0, # a_4
        0, # a_5
        0, # a_6
        0, # a_7
        0, # a_8
        0, # a_9
        0, # a_10
        0, # a_11
        0, # a_12
        0, # a_13
        0, # a_14
        0, # a_15
        0, # a_16
        0, # a_17
        0, # a_18
        # 0, # a_19
        # 0, # a_20
        # 0, # a_21
        0, # a_22
        0, # a_23
        0, # a_24
        0, # a_25
        0, # a_26
        0, # a_27
        0, # a_28
        0, # a_29
        # 0, # a_30
        # 0, # a_31
        # 0, # a_32
        # 0, # a_33
        # 0, # a_34
        # 0, # a_35
        # 0, # a_36
        # 0, # a_37
        # 0, # a_38
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
            if ambient_temperatures[index] < 15
            and collector_input_temperatures[index] < 20
        ],
        initial_guesses,
        # bounds=([-35] * 7 + [0] + [-35] * 10 + [0] + [-35] * 10 + [0] + [-35] * 3, 35),
        bounds=([-35] * 7 + [0] + [-35] * 10 + [0] + [-35] * 7 + [0], 35),
        maxfev=10000,
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
            if ambient_temperatures[index] < 15
            and collector_input_temperatures[index] < 20
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
            if 15 <= ambient_temperatures[index] < 30
            and collector_input_temperatures[index] < 20
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
            if 15 <= ambient_temperatures[index] < 30
            and collector_input_temperatures[index] < 20
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
            if 30 <= ambient_temperatures[index] <= 45
            and collector_input_temperatures[index] < 20
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
            if 30 <= ambient_temperatures[index] <= 45
            and collector_input_temperatures[index] < 20
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
            if ambient_temperatures[index] < 15
            and 20 <= collector_input_temperatures[index] < 40
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
            if ambient_temperatures[index] < 15
            and 20 <= collector_input_temperatures[index] < 40
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
            if 15 <= ambient_temperatures[index] < 30
            and 20 <= collector_input_temperatures[index] < 40
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
            if 15 <= ambient_temperatures[index] < 30
            and 20 <= collector_input_temperatures[index] < 40
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
            if 30 <= ambient_temperatures[index] <= 45
            and 20 <= collector_input_temperatures[index] < 40
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
            if 30 <= ambient_temperatures[index] <= 45
            and 20 <= collector_input_temperatures[index] < 40
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
            if ambient_temperatures[index] < 15
            and 40 <= collector_input_temperatures[index] < 60
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
            if ambient_temperatures[index] < 15
            and 40 <= collector_input_temperatures[index] < 60
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
            if 15 <= ambient_temperatures[index] < 30
            and 40 <= collector_input_temperatures[index] < 60
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
            if 15 <= ambient_temperatures[index] < 30
            and 40 <= collector_input_temperatures[index] < 60
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for ninth chunk ........... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    ninth_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_ninth_chunk,
            collector_input_temperatures_ninth_chunk,
            mass_flow_rates_ninth_chunk,
            solar_irradiances_ninth_chunk,
            wind_speeds_ninth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 30 <= ambient_temperatures[index] <= 45
            and 40 <= collector_input_temperatures[index] < 60
        ],
        initial_guesses,
    )

    ninth_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_ninth_chunk,
            collector_input_temperatures_ninth_chunk,
            mass_flow_rates_ninth_chunk,
            solar_irradiances_ninth_chunk,
            wind_speeds_ninth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 30 <= ambient_temperatures[index] <= 45
            and 40 <= collector_input_temperatures[index] < 60
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for tenth chunk ........... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    tenth_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_tenth_chunk,
            collector_input_temperatures_tenth_chunk,
            mass_flow_rates_tenth_chunk,
            solar_irradiances_tenth_chunk,
            wind_speeds_tenth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if ambient_temperatures[index] < 15
            and 60 <= collector_input_temperatures[index] < 80
        ],
        initial_guesses,
    )

    tenth_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_tenth_chunk,
            collector_input_temperatures_tenth_chunk,
            mass_flow_rates_tenth_chunk,
            solar_irradiances_tenth_chunk,
            wind_speeds_tenth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if ambient_temperatures[index] < 15
            and 60 <= collector_input_temperatures[index] < 80
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for eleventh chunk ........ ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    eleventh_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_eleventh_chunk,
            collector_input_temperatures_eleventh_chunk,
            mass_flow_rates_eleventh_chunk,
            solar_irradiances_eleventh_chunk,
            wind_speeds_eleventh_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 15 <= ambient_temperatures[index] < 30
            and 60 <= collector_input_temperatures[index] < 80
        ],
        initial_guesses,
    )

    eleventh_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_eleventh_chunk,
            collector_input_temperatures_eleventh_chunk,
            mass_flow_rates_eleventh_chunk,
            solar_irradiances_eleventh_chunk,
            wind_speeds_eleventh_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 15 <= ambient_temperatures[index] < 30
            and 60 <= collector_input_temperatures[index] < 80
        ],
        initial_guesses,
    )
    print("[  DONE  ]")

    print("Computing fit for twelth chunk .......... ", end="")
    # Attempt a curve fit based on ambient temperature chunks.
    twelth_thermal_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_twelth_chunk,
            collector_input_temperatures_twelth_chunk,
            mass_flow_rates_twelth_chunk,
            solar_irradiances_twelth_chunk,
            wind_speeds_twelth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(thermal_efficiencies)
            if 30 <= ambient_temperatures[index] <= 45
            and 60 <= collector_input_temperatures[index] < 80
        ],
        initial_guesses,
    )

    twelth_electrical_efficiency_results = curve_fit(
        _best_guess,
        (
            ambient_temperatures_twelth_chunk,
            collector_input_temperatures_twelth_chunk,
            mass_flow_rates_twelth_chunk,
            solar_irradiances_twelth_chunk,
            wind_speeds_twelth_chunk,
        ),
        [
            entry
            for index, entry in enumerate(electrical_efficiencies)
            if 30 <= ambient_temperatures[index] <= 45
            and 60 <= collector_input_temperatures[index] < 80
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
            ninth_thermal_efficiency_results,
            tenth_thermal_efficiency_results,
            eleventh_thermal_efficiency_results,
            twelth_thermal_efficiency_results,
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
            ninth_electrical_efficiency_results,
            tenth_electrical_efficiency_results,
            eleventh_electrical_efficiency_results,
            twelth_electrical_efficiency_results,
        ],
        solar_irradiances,
        electrical_efficiencies,
        wind_speeds,
    )


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
        and entry[COLLECTOR_INPUT_TEMPERATURE] < 80
        and entry[COLLECTOR_INPUT_TEMPERATURE] >= 5
        and entry[AMBIENT_TEMPERATURE] <= 45
        and entry[SOLAR_IRRADIANCE] > 0
        and entry[MASS_FLOW_RATE] > 3 / 3600
        and (
            (-1 <= entry[THERMAL_EFFICIENCY] <= 2)
            if THERMAL_EFFICIENCY in entry
            else False
        )
    ]

    ambient_temperatures = [entry[0] for entry in processed_data]
    collector_input_temperatures = [entry[1] for entry in processed_data]
    electrical_efficiencies = [entry[2] for entry in processed_data]
    mass_flow_rates = [3600 * entry[3] for entry in processed_data]
    solar_irradiances = [entry[4] for entry in processed_data]
    thermal_efficiencies = [entry[5] for entry in processed_data]
    wind_speeds = [entry[6] for entry in processed_data]

    _partial_fit(
        ambient_temperatures,
        collector_input_temperatures,
        electrical_efficiencies,
        "Mass flow rate > 3 litres/hour",
        mass_flow_rates,
        solar_irradiances,
        thermal_efficiencies,
        wind_speeds,
    )

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
        and entry[COLLECTOR_INPUT_TEMPERATURE] < 80
        and entry[COLLECTOR_INPUT_TEMPERATURE] >= 5
        and entry[AMBIENT_TEMPERATURE] <= 45
        and entry[SOLAR_IRRADIANCE] > 0
        and (entry[MASS_FLOW_RATE] <= 3 / 3600)
        and (
            (-1 <= entry[THERMAL_EFFICIENCY] <= 2)
            if THERMAL_EFFICIENCY in entry
            else False
        )
    ]

    ambient_temperatures = [entry[0] for entry in processed_data]
    collector_input_temperatures = [entry[1] for entry in processed_data]
    electrical_efficiencies = [entry[2] for entry in processed_data]
    mass_flow_rates = [3600 * entry[3] for entry in processed_data]
    solar_irradiances = [entry[4] for entry in processed_data]
    thermal_efficiencies = [entry[5] for entry in processed_data]
    wind_speeds = [entry[6] for entry in processed_data]

    _partial_fit(
        ambient_temperatures,
        collector_input_temperatures,
        electrical_efficiencies,
        "Mass flow rate <= 3 litres/hour",
        mass_flow_rates,
        solar_irradiances,
        thermal_efficiencies,
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
