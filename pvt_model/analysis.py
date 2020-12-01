#!/usr/bin/python3
"""
Does some analysis.

"""

import os

from typing import Any, Dict

import json

from matplotlib import pyplot as plt

# The first day to include in the output graph.
FIRST_DAY: int = 5
# The number of days to include in the output graph.
NUM_DAYS: int = 1
# Whether to average the data.
AVERAGE: bool = False
# The number of steps per day
STEPS_PER_DAY = 48


def load_data(filename: str) -> Dict[Any, Any]:
    """Loads some data."""
    with open(filename) as f:
        return json.load(f)


def plot(
    label: str,
    y_label: str,
    resolution: float,
    data: Dict[Any, Any],
    hold=False,
    axes=None,
    shape: str = "x",
    colour: str = None,
) -> None:
    """
    Plots something.

    :param resolution:
        The resolution of the simulation in minutes.

    :para hold:
        Whether to hold the figure and include multiple plots.

    """

    if not AVERAGE:
        x_data, y_data = (
            list(
                list(data.keys())[
                    (FIRST_DAY * STEPS_PER_DAY - 1) : (FIRST_DAY + NUM_DAYS)
                    * STEPS_PER_DAY
                ]
            ),
            list([value[label] for value in data.values()])[
                (FIRST_DAY * STEPS_PER_DAY - 1) : (FIRST_DAY + NUM_DAYS) * STEPS_PER_DAY
            ],
        )
    else:
        # Extract all data
        x_data, y_raw = (
            list(data.keys()),
            list([value[label] for value in data.values()]),
        )
        # Construct averages
        x_data = x_data[:STEPS_PER_DAY]
        y_data: list() = []
        for _ in range(STEPS_PER_DAY):
            y_data.append(sum(y_raw[::STEPS_PER_DAY]) / NUM_DAYS)
            y_raw.pop(0)
    if not hold:
        plt.clf()
    if axes is None:
        plt.scatter(x_data, y_data, label=label, marker=shape)
        (line,) = plt.plot(x_data, y_data, label=label, marker=shape)
    else:
        axes.scatter(x_data, y_data, label=label, marker=shape)
        if colour is None:
            (line,) = axes.plot(x_data, y_data, label=label, marker=shape)
        else:
            (line,) = axes.plot(x_data, y_data, label=label, marker=shape, color=colour)

    plt.xlabel("Time / Hour")
    plt.ylabel(y_label)
    return line


def save_figure(figure_name: str) -> None:
    """
    Saves the figure, shuffling existing files out of the way.

    :param figure_name:
        The name of the figure to save.

    """

    # Determine the maximum figure that already exists.
    figure_int = 1
    while os.path.exists("figure_{}_{}.jpg".format(figure_name, figure_int)):
        figure_int += 1
    # Shuffle all figures along
    while figure_int >= 2:
        os.rename(
            "figure_{}_{}.jpg".format(figure_name, figure_int - 1),
            "figure_{}_{}.jpg".format(figure_name, figure_int),
        )
        figure_int -= 1
    # Save the figure
    plt.savefig("figure_{}_1.jpg".format(figure_name))


if __name__ == "__main__":
    data = load_data("data_output.json")

    things_to_plot = [
        "collector_temperature",
        "collector_temperature_gain",
        "tank_temperature",
        "tank_output_temperature",
        "ambient_temperature",
        "sky_temperature",
    ]  # "collector_temperature", "tank_temperature"]
    lines = [
        plot(thing, "Temperature / degC", 30, data, hold=True)
        for thing in things_to_plot
    ]
    plt.legend(lines, things_to_plot)
    save_figure("tank_temperature")

    plt.clf()

    things_to_plot = [
        "glass_temperature",
        "pv_temperature",
        "collector_temperature",
        "ambient_temperature",
        "sky_temperature",
    ]  # "collector_temperature", "tank_temperature"]
    lines = [
        plot(thing, "Temperature / degC", 30, data, hold=True)
        for thing in things_to_plot
    ]
    plt.legend(lines, things_to_plot)

    save_figure("temperature")

    plt.clf()

    fig, ax1 = plt.subplots()
    dc_to_plot = ["dc_electrical", "dc_thermal"]
    lines = [
        plot(thing, "Demand Covered / %", 30, data, hold=True, axes=ax1)
        for thing in dc_to_plot
    ]
    plt.legend(lines, dc_to_plot)

    demand_to_plot = ["thermal_load"]
    ax2 = ax1.twinx()
    lines = [
        plot(
            thing,
            "Thermal Demand / litres",
            30,
            data,
            hold=True,
            axes=ax2,
            shape="+",
            colour="purple",
        )
        for thing in demand_to_plot
    ]
    plt.legend(lines, demand_to_plot)

    save_figure("dc")

    plt.clf()
    other_to_plot = ["solar_irradiance", "normal_irradiance"]
    lines = [
        plot(thing, "Solar Irradiance / Watts / meter squared", 30, data, hold=True)
        for thing in other_to_plot
    ]
    plt.legend(lines, other_to_plot)

    save_figure("solar_irradiance")
