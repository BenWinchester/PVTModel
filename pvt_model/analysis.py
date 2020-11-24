#!/usr/bin/python3
"""
Does some analysis.

"""

import os

from typing import Any, Dict

import json

from matplotlib import pyplot as plt


def load_data(filename: str) -> Dict[Any, Any]:
    """Loads some data."""
    with open(filename) as f:
        return json.load(f)


def plot(
    label: str, y_label: str, resolution: float, data: Dict[Any, Any], hold=False
) -> None:
    """
    Plots something.

    :param resolution:
        The resolution of the simulation in minutes.

    :para hold:
        Whether to hold the figure and include multiple plots.

    """

    x_data, y_data = (
        list(data.keys()),
        list([value[label] for value in data.values()]),
    )
    if not hold:
        plt.clf()
    plt.scatter(x_data, y_data, label=label)
    (line,) = plt.plot(x_data, y_data, label=label)
    plt.xlabel("Time / Hour")
    plt.ylabel(y_label)
    return line


if __name__ == "__main__":
    data = load_data("data_output.json")
    things_to_plot = [
        "glass_temperature",
        "pv_temperature",
        "tank_temperature",
        "collector_temperature",
        "collector_output_temperature",
    ]  # "collector_temperature", "tank_temperature"]
    lines = [
        plot(thing, "Temperature / degC", 30, data, hold=True)
        for thing in things_to_plot
    ]
    plt.legend(lines, things_to_plot)
    figure_int = 1
    while os.path.exists("figure_temperature_{}.jpg".format(figure_int)):
        figure_int += 1
    plt.savefig("figure_temperature_{}.jpg".format(figure_int))

    plt.clf()
    dc_to_plot = ["dc_electrical", "dc_thermal"]
    lines = [
        plot(thing, "Demand Covered / %", 30, data, hold=True) for thing in dc_to_plot
    ]
    plt.legend(lines, dc_to_plot)

    figure_int = 1
    while os.path.exists("figure_dc_{}.jpg".format(figure_int)):
        figure_int += 1
    plt.savefig("figure_dc_{}.jpg".format(figure_int))
