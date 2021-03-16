#!/usr/bin/python3.7
# type: ignore
########################################################################################
# analysis.py - The analysis component for the model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################
"""
Used for analysis of the output of the model runs.

NOTE: The mypy type checker is instructed to ignore this component. This is done due to
the lower standards applied to the analysis code, and the failure of mypy to correctly
type-check the external matplotlib.pyplot module.

"""

import argparse
import os
import sys

from logging import Logger
from typing import Any, List, Dict, Optional, Tuple, Union

import json
import numpy
import re

from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3D

try:
    from ..__utils__ import get_logger
    from ..pvt_system_model.constants import (  # pylint: disable=unused-import
        HEAT_CAPACITY_OF_WATER,
    )
    from .__utils__ import GraphDetail
except ModuleNotFoundError:
    import logging

    logging.error(
        "Incorrect module import. Try running with `python3.7 -m pvt_model.analysis`"
    )
    raise

__all__ = ("analyse",)

# The directory into which which should be saved
NEW_FIGURES_DIRECTORY: str = "figures"
# The directory in which old figures are saved and stored for long-term access
OLD_FIGURES_DIRECTORY: str = "old_figures"
# How detailed the graph should be
GRAPH_DETAIL: GraphDetail = GraphDetail.lowest
# How many values there should be between each tick on the x-axis
X_TICK_SEPARATION: int = int(8 * GRAPH_DETAIL.value / 48)
# Which days of data to include
DAYS_TO_INCLUDE: List[bool] = [False, True]


def _parse_args(args) -> argparse.Namespace:
    """
    Parse the CLI args.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-file-name", "-df", help="Path to the data file to parse."
    )
    parser.add_argument(
        "--show-output",
        "-so",
        action="store_true",
        default=False,
        help="Show the output figures generated.",
    )

    return parser.parse_args(args)


def _resolution_from_graph_detail(
    graph_detail: GraphDetail, num_data_points: int
) -> int:
    """
    Determine the x-step resolution of the plot.

    I realise here that the word "resolution" is over-used. Here, the number of data
    points (in the system data) that need to be absorbed into one graph point is
    calculated and returned.

    :param graph_detail:
        The detail level needed on the graph.

    :param num_data_points:
        The number of data points recorded in the plot.

    :return:
        The number of data points that need to be absorbed per graph point.

    """

    # * For "lowest", include one point every half-hour.
    if graph_detail == GraphDetail.highest:
        return 1

    return int(num_data_points / graph_detail.value)


def _reduce_data(
    data_to_reduce: Dict[str, Dict[Any, Any]],
    graph_detail: GraphDetail,
    logger: Logger,
) -> Dict[Union[int, str], Dict[Any, Any]]:
    """
    This processes the data, using sums to reduce the resolution so it can be plotted.

    :param data_to_reduce:
        The raw, JSON data, contained within a `dict`.

    :param graph_detail:
        The level of detail required in the graph.

    :param logger:
        The logger being used in the run.

    :return:
        The cropped/summed up data, returned at a lower resolution as specified by the
        graph detail.

    """

    # Determine the number of data points to be amalgamated per graph point.
    data_points_per_graph_point: int = _resolution_from_graph_detail(
        graph_detail, len(data_to_reduce)
    )

    if data_points_per_graph_point <= 1:
        return data_to_reduce

    # Construct a dictionary to contain this reduced data.
    reduced_data: Dict[Union[int, str], Dict[Any, Any]] = {
        index: dict()
        for index in range(
            int(
                len([key for key in data_to_reduce if key.isdigit()])
                / data_points_per_graph_point
            )
        )
    }

    # Depending on the type of data entry, i.e., whether it is a temperature, load,
    # demand covered, or irradiance (or other), the way that it is processed will vary.
    for data_entry_name in data_to_reduce["0"].keys():
        # pdb.set_trace(header="Beginning of reduction loop.")
        # * If the entry is a date or time, just take the value
        if data_entry_name in ["date", "time"]:
            for index, _ in enumerate(reduced_data):
                reduced_data[index][data_entry_name] = data_to_reduce[
                    str(index * data_points_per_graph_point)
                ][data_entry_name]
            continue

        # * If the data entry is a temperature, or a power output, then take a rolling
        # average
        if any(
            [
                key in data_entry_name
                for key in [
                    "temperature",
                    "irradiance",
                    "efficiency",
                    "hot_water_load",
                    "electrical_load",
                ]
            ]
        ):
            for outer_index, _ in enumerate(reduced_data):
                # Attempt to process as a dict or float first.
                if isinstance(data_to_reduce["0"][data_entry_name], float):
                    reduced_data[outer_index][data_entry_name] = sum(
                        [
                            float(data_to_reduce[str(inner_index)][data_entry_name])
                            / data_points_per_graph_point
                            for inner_index in range(
                                int(data_points_per_graph_point * outer_index),
                                int(data_points_per_graph_point * (outer_index + 1)),
                            )
                        ]
                    )
                elif isinstance(data_to_reduce["0"][data_entry_name], dict):
                    # Loop through the various coordinate pairs.
                    for sub_dict_key in data_to_reduce[str(outer_index)][
                        data_entry_name
                    ]:
                        if data_entry_name not in reduced_data[outer_index]:
                            reduced_data[outer_index][data_entry_name]: Dict[
                                str, float
                            ] = dict()
                        reduced_data[outer_index][data_entry_name][sub_dict_key] = sum(
                            [
                                float(
                                    data_to_reduce[str(inner_index)][data_entry_name][
                                        sub_dict_key
                                    ]
                                )
                                / data_points_per_graph_point
                                for inner_index in range(
                                    int(data_points_per_graph_point * outer_index),
                                    int(
                                        data_points_per_graph_point * (outer_index + 1)
                                    ),
                                )
                            ]
                        )
                else:
                    logger.debug(
                        "A value was an unsuported type. Setting to 'None': %s",
                        data_entry_name,
                    )
                    for index, _ in enumerate(reduced_data):
                        reduced_data[index][data_entry_name] = None
                    continue

        # * If the data entry is a load, then take a sum
        elif any([key in data_entry_name for key in ["load", "output"]]):
            # @@@
            # * Here, the data is divided by 3600 to convert from Joules to Watt Hours.
            # * This only works provided that we are dealing with values in Joules...
            for outer_index, _ in enumerate(reduced_data):
                reduced_data[outer_index][data_entry_name] = (
                    sum(
                        [
                            float(data_to_reduce[str(inner_index)][data_entry_name])
                            for inner_index in range(
                                int(data_points_per_graph_point * outer_index),
                                int(data_points_per_graph_point * (outer_index + 1)),
                            )
                        ]
                    )
                    / 3600
                )
            continue

        # * Otherwise, we just use the data point.
        for index, _ in enumerate(reduced_data):
            reduced_data[index][data_entry_name] = data_to_reduce[
                str(index * data_points_per_graph_point)
            ][data_entry_name]
        continue

    return reduced_data


def _post_process_data(
    data_to_post_process: Dict[str, Dict[Any, Any]]
) -> Dict[str, Dict[Any, Any]]:
    """
    Carries out post-processing on data where necessary.

    Things to be computed:
        - Bulk Water Temperautre
        - Litres consumed

    :param data_to_post_process:
        The data to be post processed.

    :return:
        The post-processed data.

    """

    # * Cycle through all the data points and compute the new values as needed.
    for data_entry in data_to_post_process.values():
        data_entry["collector_temperature_gain"] = (
            data_entry["collector_output_temperature"]
            - data_entry["collector_input_temperature"]
        )
    #     # Conversion needed from Wh to Joules.
    #     data_entry["litres_per_hour"] = (
    #         data_entry["thermal_load"] / (HEAT_CAPACITY_OF_WATER * 50) * 60
    #     )
    return data_to_post_process


def load_model_data(filename: str) -> Dict[Any, Any]:
    """
    Loads some model_data that was generated by the model.

    :param filename:
        The name of the model_data file to open.

    :return:
        The JSON model_data, loaded as a `dict`.

    """

    with open(filename, "r") as f:
        json_data: Dict[Any, Any] = json.load(f)

    return json_data


def _annotate_maximum(
    model_data: Dict[Any, Any], y_axis_labels: List[str], axis
) -> None:
    """
    Annotates the maximum value on a plot.

    .. citation:: Taken with permission from:
    https://stackoverflow.com/questions/43374920/how-to-automatically-annotate-maximum-value-in-pyplot/43375405

    :param data:
        The model data to find the maxima from.

    :param y_axis_labels:
        Labels for the y axis to process.

    :param axis:
        The axis object, if relevant.

    """

    # * For each series, determine the maximum data points:
    box_text = ""
    x_series = [data_entry["time"] for data_entry in model_data.values()]
    for y_lab in y_axis_labels:
        y_series = [data_entry[y_lab] for data_entry in model_data.values()]
        x_max = x_series[max(enumerate(y_series))[0]]
        y_max = max(y_series)
        box_text += f"max({y_lab})={y_max:.2f} at {x_max}\n"
    box_text.strip()

    # Fetch the axis if necessary.
    if not axis:
        axis = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=85")
    kwargs = dict(
        xycoords="data",
        textcoords="axes fraction",
        arrowprops=arrowprops,
        bbox=bbox_props,
        ha="right",
        va="top",
    )
    axis.ticklabel_format(useOffset=False)
    axis.annotate(box_text, xy=(x_max, y_max), xytext=(0.8, 0.8), **kwargs)


def plot(
    label: str,
    y_label: str,
    model_data: Dict[Any, Any],
    hold=False,
    axes=None,
    shape: str = "x",
    colour: str = None,
    bar_plot: bool = False,
) -> Optional[Any]:
    """
    Plots some model_data based on input parameters.

    :param label:
        The name of the model_data label to plot. For example, this could be "pv_temperature",
        in which case, the pv temperature would be plotted at the tine steps specified
        in the model.

    :param y_label:
        The label to assign to the y axis when plotting.

    :param resolution:
        The resolution of the model that was run. This is measured in minutes.

    :param model_data:
        The model_data, loaded from the model's output file.

    :param axes:
        If provided, a separate axis is used for plotting the model_data.

    :param hold:
        Whether to hold the screen between plots (True) or reset it (False).

    :param shape:
        This sets the shape of the marker for `matplotlib.pyplot` to use when plotting
        the model_data.

    :param colour:
        The colour to use for the plot.

    :param bar_plot:
        Whether to plot a line graph (False) or a bar_plot plot (True).

    """

    x_model_data, y_model_data = (
        [entry["time"] for entry in model_data.values()],
        [entry[label] for entry in model_data.values()],
    )

    # If we are not holding the graph, then clear the model_data.
    if not hold:
        plt.clf()

    # Reduce the values on the x axis to be times.
    # x_model_data = [float(item) / (resolution / 60) for item in x_model_data]

    if bar_plot:
        if axes is None:
            if colour is None:
                plt.bar(x_model_data, y_model_data, label=label)
            else:
                plt.bar(x_model_data, y_model_data, label=label, color=colour)

        #  otherwise, the model_data needs to be plotted on just on axis.
        else:
            if colour is None:
                line = axes.bar(x_model_data, y_model_data, label=label)
            else:
                line = axes.bar(x_model_data, y_model_data, label=label, color=colour)

    else:
        # If we are not using axes, then the model_data can be straight plotted...
        if axes is None:
            plt.scatter(x_model_data, y_model_data, label=label, marker=shape)
            (line,) = plt.plot(x_model_data, y_model_data, label=label, marker=shape)

        #  otherwise, the model_data needs to be plotted on just on axis.
        else:
            axes.scatter(x_model_data, y_model_data, label=label, marker=shape)
            if colour is None:
                (line,) = axes.plot(
                    x_model_data, y_model_data, label=label, marker=shape
                )
            else:
                (line,) = axes.plot(
                    x_model_data, y_model_data, label=label, marker=shape, color=colour
                )

    # Set the labels for the axes.
    plt.xlabel("Time of Day")
    plt.ylabel(y_label)

    return line


def save_figure(figure_name: str) -> None:
    """
    Saves the figure, shuffling existing files out of the way.

    :param figure_name:
        The name of the figure to save.

    """

    # Create a regex for cycling through the files.
    file_regex = re.compile("figure_{}_(?P<old_index>[0-9]).jpg".format(figure_name))

    # Create a storage directory if it doesn't already exist.
    if not os.path.isdir(OLD_FIGURES_DIRECTORY):
        os.mkdir(OLD_FIGURES_DIRECTORY)
    if not os.path.isdir(NEW_FIGURES_DIRECTORY):
        os.mkdir(NEW_FIGURES_DIRECTORY)

    # We need to work download from large numbers to new numbers.
    filenames = sorted(os.listdir(OLD_FIGURES_DIRECTORY))
    filenames.reverse()

    # Incriment all files in the old_figures directory.
    for filename in filenames:
        file_match = re.match(file_regex, filename)
        if file_match is None:
            continue
        new_file_name = re.sub(
            str(file_match.group("old_index")),
            str(int(file_match.group("old_index")) + 1),
            filename,
        )
        os.rename(
            os.path.join(OLD_FIGURES_DIRECTORY, filename),
            os.path.join(OLD_FIGURES_DIRECTORY, new_file_name),
        )

    # Move the current _1 file into the old directory
    if os.path.isfile(
        os.path.join(NEW_FIGURES_DIRECTORY, f"figure_{figure_name}_1.jpg")
    ):
        os.rename(
            os.path.join(NEW_FIGURES_DIRECTORY, f"figure_{figure_name}_1.jpg"),
            os.path.join(OLD_FIGURES_DIRECTORY, f"figure_{figure_name}_1.jpg"),
        )

    # If it exists, move the current figure to _1
    if os.path.isfile(os.path.join(NEW_FIGURES_DIRECTORY, f"figure_{figure_name}.jpg")):
        os.rename(
            os.path.join(NEW_FIGURES_DIRECTORY, f"figure_{figure_name}.jpg"),
            os.path.join(NEW_FIGURES_DIRECTORY, f"figure_{figure_name}_1.jpg"),
        )

    # Save the figure
    plt.savefig(os.path.join(NEW_FIGURES_DIRECTORY, f"figure_{figure_name}.jpg"))


def plot_figure(
    figure_name: str,
    model_data: Dict[Any, Any],
    first_axis_things_to_plot: List[str],
    first_axis_label: str,
    *,
    first_axis_shape: str = "x",
    first_axis_y_limits: Optional[Tuple[int, int]] = None,
    second_axis_things_to_plot: Optional[List[str]] = None,
    second_axis_label: Optional[str] = None,
    second_axis_y_limits: Optional[Tuple[int, int]] = None,
    annotate_maximum: bool = False,
    bar_plot: bool = False,
) -> None:
    """
    Does all the work needed to plot a figure with up to two axes and save it.

    :param figure_name:
        The name to assign to the figure when saving it.

    :param model_data:
        The model_data, as extracted from the JSON outputted by the simulation.

    :param first_axis_things_to_plot:
        The list of variable names (keys in the JSON model_data) to plot on the first axis.

    :param first_axis_label:
        The label to assign to the first y-axis.

    :param first_axis_y_limits:
        A `tuple` giving the lower and upper limits to set for the y axis for the first
        axis.

    :param first_axis_shape:
        A `str` giving an optional override shape for the first axis.

    :param second_axis_things_to_plot:
        The list of variable names (keys in the JSON model_data) to plot on the second axis.

    :param second_axis_label:
        The label to assign to the second y-axis.

    :param second_axis_y_limits:
        A `tuple` giving the lower and upper limits to set for the y axis for the second
        axis.

    :param annotate_maximum:
        If specified, the maximum will be plotted on the graph.

    :param bar_plot:
        If specified, a bar plot will be generated, rather than a line plot.

    """

    _, ax1 = plt.subplots()

    lines = [
        plot(
            entry,
            first_axis_label,
            model_data,
            hold=True,
            axes=ax1,
            shape=first_axis_shape,
            bar_plot=bar_plot,
        )
        for entry in first_axis_things_to_plot
    ]

    ax1.set_xticks(ax1.get_xticks()[::X_TICK_SEPARATION])

    # Set the y limits if appropriate
    if first_axis_y_limits is not None:
        plt.ylim(*first_axis_y_limits)

    # Plot a maximum value box if requested
    if annotate_maximum:
        _annotate_maximum(
            model_data,
            first_axis_things_to_plot,
            ax1,
        )

    # Save the figure and return if only one axis is plotted.
    if second_axis_things_to_plot is None:
        plt.legend(lines, first_axis_things_to_plot)  # , loc="upper left")
        save_figure(figure_name)
        return

    # Second-axis plotting.
    ax2 = ax1.twinx()

    lines.extend(
        [
            plot(
                entry,
                second_axis_label,
                model_data,
                hold=True,
                axes=ax2,
                shape=".",
                bar_plot=bar_plot,
            )
            for entry in second_axis_things_to_plot
        ]
    )

    plt.legend(lines, first_axis_things_to_plot + second_axis_things_to_plot)

    ax2.set_xticks(ax2.get_xticks()[::X_TICK_SEPARATION])

    # Set the y limits if appropriate.
    if second_axis_y_limits is not None:
        plt.ylim(*second_axis_y_limits)

    # Plot a maximum value box if requested
    if annotate_maximum:
        _annotate_maximum(
            model_data,
            second_axis_things_to_plot,
            ax2,
        )

    save_figure(figure_name)


def plot_two_dimensional_figure(
    figure_name: str,
    logger: Logger,
    model_data: Dict[Any, Any],
    thing_to_plot: str,
    *,
    axis_label: str,
    hour: int,
    minute: int,
    hold: bool = False,
) -> None:
    """
    Plots a two-dimensional figure.

    :param figure_name:
        The name to use when saving the figure.

    :param loger:
        The logger used for the analysis module.

    :param model_data:
        The data outputted from the model.

    :param thing_to_plot:
        The name of the variable to plot.

    :param axis_label:
        The label for the y-axis of the plot.

    :param hour:
        The hour at which to plot the two-dimensional temperature profile.

    :param minute:
        The minute at which to plot the two-dimensional temperature profile.

    :param hold:
        Whether to hold the plot.

    """

    # Determine the data index number based on the time.
    entry_number = int(len(model_data) * ((hour / 24) + (minute / (24 * 60))))
    try:
        data_entry = model_data[entry_number]
    except KeyError:
        try:
            data_entry = model_data[str(entry_number)]
        except KeyError as e:
            logger.error(
                "Key lookup failed for data entry number %s: %s", entry_number, str(e)
            )
            raise

    if thing_to_plot not in data_entry:
        logger.error(
            "Unable to plot %s. Either data is not 2D or name mismatch occured in the "
            "analysis module. See /logs for details.",
            thing_to_plot,
        )
        return

    coordinate_regex = re.compile(r"\((?P<x_index>[0-9]*), (?P<y_index>[0-9]*)\)")

    try:
        x_series = [
            int(re.match(coordinate_regex, coordinate).group("x_index"))
            for coordinate in data_entry[thing_to_plot]
        ]
        y_series = [
            int(re.match(coordinate_regex, coordinate).group("y_index"))
            for coordinate in data_entry[thing_to_plot]
        ]
        z_series = list(data_entry[thing_to_plot].values())
    except AttributeError as e:
        logger.info(str(e))
        logger.error(
            "Unable to match coordinate output for %s. Check output file contents. See "
            "/logs for details.",
            thing_to_plot,
        )
        return

    # Reshape the data for plotting.
    array_shape = (len(set(x_series)), len(set(y_series)))

    # If the data is only 1D, then plot a standard 1D profile through the collector.
    if 1 in array_shape:
        # If we are not holding the graph, then clear the model_data.
        if not hold:
            plt.clf()
        lines = plt.plot(y_series, z_series)
        # Set the axis limits to be sensible.
        plt.ylim(0, 100)
        # Set the labels for the axes.
        plt.xlabel("Segment y index")
        plt.ylabel(axis_label)
        # Add the legend.
        plt.legend(lines, [thing_to_plot])
        save_figure(figure_name)

        return

    x_array = numpy.reshape(x_series, array_shape)
    y_array = numpy.reshape(y_series, array_shape)
    z_array = numpy.reshape(z_series, array_shape)

    # Plot the figure.
    fig3D = plt.figure()
    axes3D = fig3D.gca(projection="3d")
    surface = plt3D.plot_surface(
        axes3D,
        x_array,
        y_array,
        z_array,
        cmap=cm.coolwarm,  # pylint: disable=no-member
        linewidth=0,
        antialiased=False,
    )

    # Add axes and colour scale.
    fig3D.colorbar(surface, shrink=0.5, aspect=5)
    axes3D.set_xlabel("X index")
    axes3D.set_ylabel("Y index")
    axes3D.set_zlabel(axis_label)

    save_figure(figure_name)


def analyse(data_file_name: str, show_output: Optional[bool] = False) -> None:
    """
    The main method for the analysis module.

    :param data_file_name:
        The path to the data file to analyse.

    :param show_output:
        Whether to show the output files generated.

    """

    # * Set up the logger
    logger = get_logger("pvt_analysis", True)

    # * Extract the data.
    data = load_model_data(data_file_name)

    # * Reduce the resolution of the data.
    data = _reduce_data(data, GRAPH_DETAIL, logger)

    # * Create new data values where needed.
    data = _post_process_data(data)

    # Plot All Temperatures
    plot_figure(
        "all_temperatures",
        data,
        first_axis_things_to_plot=[
            "ambient_temperature",
            "bulk_water_temperature",
            "collector_temperature",
            "collector_input_temperature",
            "collector_output_temperature",
            "glass_temperature",
            "pipe_temperature",
            "pv_temperature",
            "sky_temperature",
            "tank_temperature",
        ],
        first_axis_label="Temperature / deg C",
        first_axis_y_limits=[-10, 110],
    )

    # Plot All Temperatures
    plot_figure(
        "all_temperatures_unbounded",
        data,
        first_axis_things_to_plot=[
            "ambient_temperature",
            "bulk_water_temperature",
            "collector_temperature",
            "collector_input_temperature",
            "collector_output_temperature",
            "glass_temperature",
            "pipe_temperature",
            "pv_temperature",
            "sky_temperature",
            "tank_temperature",
        ],
        first_axis_label="Temperature / deg C",
    )

    # # Plot Figure 4a: Electrical Demand
    # plot_figure(
    #     "maria_4a_electrical_load",
    #     data,
    #     ["electrical_load"],
    #     "Dwelling Load Profile / W",
    #     first_axis_y_limits=[0, 5000],
    #     first_axis_shape="d",
    # )

    # # Plot Figure 4b: Thermal Demand
    # plot_figure(
    #     "maria_4b_thermal_load",
    #     data,
    #     ["hot_water_load"],
    #     "Hot Water Consumption / Litres per hour",
    #     first_axis_y_limits=[0, 12],
    #     bar_plot=True,
    # )

    # # Plot Figure 5a: Diurnal Solar Irradiance
    # plot_figure(
    #     "maria_5a_solar_irradiance",
    #     data,
    #     [
    #         "solar_irradiance",
    #         # "normal_irradiance"
    #     ],
    #     "Solar Irradiance / Watts / meter squared",
    #     first_axis_y_limits=[0, 600],
    # )

    # Plot Figure 5b: Ambient Temperature
    plot_figure(
        "maria_5b_ambient_temperature",
        data,
        first_axis_things_to_plot=["ambient_temperature", "sky_temperature"],
        first_axis_label="Temperature / deg C",
        first_axis_y_limits=[0, 65],
    )

    # Plot Figure 6a: Panel-related Temperatures
    plot_figure(
        "maria_6a_panel_temperature",
        data,
        first_axis_things_to_plot=[
            "ambient_temperature",
            "bulk_water_temperature",
            "collector_temperature",
            "glass_temperature",
            "pipe_temperature",
            "pv_temperature",
            "sky_temperature",
        ],
        first_axis_label="Temperature / deg C",
        first_axis_y_limits=[-10, 50],
    )

    # Plot Figure 6b: Tank-related Temperatures
    plot_figure(
        "maria_6b_tank_temperature",
        data,
        first_axis_things_to_plot=[
            "collector_output_temperature",
            "collector_input_temperature",
            "tank_temperature",
        ],
        first_axis_label="Temperature / deg C",
        first_axis_y_limits=[0, 50],
    )

    # Plot Figure 7: Stream-related Temperatures
    plot_figure(
        "maria_7_stream_temperature",
        data,
        first_axis_things_to_plot=[
            "collector_temperature_gain",
            "exchanger_temperature_drop",
        ],
        first_axis_label="Temperature Gain / deg C",
        # second_axis_things_to_plot=["tank_heat_addition"],
        # second_axis_label="Tank Heat Addition / W",
    )

    # # Plot Figure 8A - Electrical Power and Net Electrical Power
    # plot_figure(
    #     "maria_8a_electrical_output",
    #     data,
    #     ["gross_electrical_output", "net_electrical_output"],
    #     "Electrical Energy Supplied / Wh",
    # )

    # # Plot Figure 8B - Thermal Power Supplied and Thermal Power Demanded
    # plot_figure(
    #     "maria_8b_thermal_output",
    #     data,
    #     ["thermal_load", "thermal_output"],
    #     "Thermal Energy Supplied / Wh",
    # )

    # # Plot Figure 10 - Electrical Power, Gross only
    # plot_figure(
    #     "maria_10_gross_electrical_output",
    #     data,
    #     ["gross_electrical_output"],
    #     "Electrical Energy Supplied / Wh",
    # )

    # Plot glass layer temperatures at midnight, 6 am, noon, and 6 pm.
    plot_two_dimensional_figure(
        "glass_temperature_0000",
        logger,
        data,
        "layer_temperature_map_glass",
        axis_label="Temperature / degC",
        hour=0,
        minute=0,
    )

    plot_two_dimensional_figure(
        "glass_temperature_0600",
        logger,
        data,
        "layer_temperature_map_glass",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
    )

    plot_two_dimensional_figure(
        "glass_temperature_1200",
        logger,
        data,
        "layer_temperature_map_glass",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
    )

    plot_two_dimensional_figure(
        "glass_temperature_1800",
        logger,
        data,
        "layer_temperature_map_glass",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
    )

    # Plot PV layer temperatures at midnight, 6 am, noon, and 6 pm.
    plot_two_dimensional_figure(
        "pv_temperature_0000",
        logger,
        data,
        "layer_temperature_map_pv",
        axis_label="Temperature / degC",
        hour=0,
        minute=0,
    )

    plot_two_dimensional_figure(
        "pv_temperature_0600",
        logger,
        data,
        "layer_temperature_map_pv",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
    )

    plot_two_dimensional_figure(
        "pv_temperature_1200",
        logger,
        data,
        "layer_temperature_map_pv",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
    )

    plot_two_dimensional_figure(
        "pv_temperature_1800",
        logger,
        data,
        "layer_temperature_map_pv",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
    )

    # Plot collector layer temperatures at midnight, 6 am, noon, and 6 pm.
    plot_two_dimensional_figure(
        "collector_temperature_0000",
        logger,
        data,
        "layer_temperature_map_collector",
        axis_label="Temperature / degC",
        hour=0,
        minute=0,
    )

    plot_two_dimensional_figure(
        "collector_temperature_0600",
        logger,
        data,
        "layer_temperature_map_collector",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
    )

    plot_two_dimensional_figure(
        "collector_temperature_1200",
        logger,
        data,
        "layer_temperature_map_collector",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
    )

    plot_two_dimensional_figure(
        "collector_temperature_1800",
        logger,
        data,
        "layer_temperature_map_collector",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
    )

    # Plot bulk water temperatures at midnight, 6 am, noon, and 6 pm.
    plot_two_dimensional_figure(
        "pipe_temperature_0000",
        logger,
        data,
        "layer_temperature_map_pipe",
        axis_label="Temperature / degC",
        hour=0,
        minute=0,
    )

    plot_two_dimensional_figure(
        "pipe_temperature_0600",
        logger,
        data,
        "layer_temperature_map_pipe",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
    )

    plot_two_dimensional_figure(
        "pipe_temperature_1200",
        logger,
        data,
        "layer_temperature_map_pipe",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
    )

    plot_two_dimensional_figure(
        "pipe_temperature_1800",
        logger,
        data,
        "layer_temperature_map_pipe",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
    )

    # Plot bulk water temperatures at midnight, 6 am, noon, and 6 pm.
    plot_two_dimensional_figure(
        "bulk_water_temperature_0000",
        logger,
        data,
        "layer_temperature_map_bulk_water",
        axis_label="Temperature / degC",
        hour=0,
        minute=0,
    )

    plot_two_dimensional_figure(
        "bulk_water_temperature_0600",
        logger,
        data,
        "layer_temperature_map_bulk_water",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
    )

    plot_two_dimensional_figure(
        "bulk_water_temperature_1200",
        logger,
        data,
        "layer_temperature_map_bulk_water",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
    )

    plot_two_dimensional_figure(
        "bulk_water_temperature_1800",
        logger,
        data,
        "layer_temperature_map_bulk_water",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
    )

    """  # pylint: disable=pointless-string-statement
    # * Plotting all tank-related temperatures
    plot_figure(
        "tank_temperature",
        data,
        [
            "collector_temperature",
            "collector_output_temperature",
            "collector_temperature_gain",
            "tank_temperature",
            # "tank_output_temperature",
            "ambient_temperature",
            "sky_temperature",
        ],
        "Temperature / degC",
    )

    # * Plotting all temperatures relevant in the system.
    plot_figure(
        "all_temperatures",
        data,
        [
            "collector_temperature",
            "collector_output_temperature",
            "collector_temperature_gain",
            "tank_temperature",
            # "tank_output_temperature",
            "ambient_temperature",
            "sky_temperature",
            "glass_temperature",
            "pv_temperature",
        ],
        "Temperature / degC",
    )

    # * Plotting all PV-T panel layer temperatures
    plot_figure(
        "pvt_panel_temperature",
        data,
        [
            "glass_temperature",
            "pv_temperature",
            "collector_temperature",
            "ambient_temperature",
            "sky_temperature",
        ],
        "Temperature / degC",
    )

    # * Plotting all temperatures in an unglazed panel
    plot_figure(
        "unglazed_pvt_temperature",
        data,
        [
            "pv_temperature",
            "collector_temperature",
            "ambient_temperature",
            "sky_temperature",
        ],
        "Temperature / degC",
    )

    # * Plotting thermal-collector-only temperatures
    plot_figure(
        "isolated_thermal_collector",
        data,
        [
            "collector_temperature",
            "ambient_temperature",
            "sky_temperature",
        ],
        "Temperature / degC",
    )

    # * Plotting demand covered and thermal load on one graph
    plot_figure(
        "demand_covered",
        data,
        first_axis_things_to_plot=["dc_electrical", "dc_thermal"],
        first_axis_label="Demand Covered / %",
        first_axis_y_limits=(0, 100),
        second_axis_things_to_plot=["thermal_load", "thermal_output"],
        second_axis_label="Thermal Energy Supplied / Wh",
    )

    # * Plotting the auxiliary heating required along with the thermal load on the
    # * system.

    plot_figure(
        "auxiliary_heating",
        data,
        first_axis_things_to_plot=["auxiliary_heating", "tank_heat_addition"],
        first_axis_label="Auxiliary Heating and Tank Heat Addition / Watts",
        second_axis_things_to_plot=["thermal_load", "thermal_output"],
        second_axis_label="Thermal Energy Supplied / Wh",
    )

    # * Plotting the collector input, output, gain, and temperature.
    plot_figure(
        "collector_temperatures",
        data,
        [
            "collector_temperature",
            "collector_output_temperature",
            "collector_input_temperature",
            "collector_temperature_gain",
            "tank_temperature",
        ],
        "Temperature / K",
    )

    # * Plotting the tank temperature, collector temperature, and heat inputted into the
    # * tank.
    plot_figure(
        "tank_heat_gain_profile",
        data,
        first_axis_things_to_plot=["tank_temperature", "collector_temperature"],
        first_axis_label="Temperature / deg C",
        first_axis_y_limits=(0, 100),
        second_axis_things_to_plot=["tank_heat_addition"],
        second_axis_label="Tank Heat Input / Watts",
    )

    # * Plotting the tank temperature, collector temperature, and heat inputted into the
    # * tank.
    """  # pylint: disable=pointless-string-statement

    if show_output:
        plt.show()


if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])
    analyse(parsed_args.data_file_name, parsed_args.show_output)
