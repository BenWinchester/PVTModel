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
import re

import numpy

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

# Used to distinguish dynamic data sets.
DYNAMIC_DATA_TYPE = "dynamic"
# The directory into which which should be saved
NEW_FIGURES_DIRECTORY: str = "figures"
# Used to identify the "time" data set.
TIME_KEY = "time"
# The directory in which old figures are saved and stored for long-term access
OLD_FIGURES_DIRECTORY: str = "old_figures"
# Used to distinguish steady-state data sets.
STEADY_STATE_DATA_TYPE = "steady_state"
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


def _reduce_data(  # pylint: disable=too-many-branches
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

    logger.info("Beginning data reduction.")

    # Determine the number of data points to be amalgamated per graph point.
    data_points_per_graph_point: int = _resolution_from_graph_detail(
        graph_detail, len(data_to_reduce)
    )

    logger.info(
        "%s data points will be condensed to each graph point.",
        data_points_per_graph_point,
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
    for data_entry_name in data_to_reduce[  # pylint: disable=too-many-nested-blocks
        "0"
    ].keys():
        # pdb.set_trace(header="Beginning of reduction loop.")
        # * If the entry is a date or time, just take the value
        if data_entry_name in ["date", TIME_KEY]:
            for index, _ in enumerate(reduced_data):
                reduced_data[index][data_entry_name] = data_to_reduce[
                    str(index * data_points_per_graph_point)
                ][data_entry_name]
            continue

        # * If the data entry is a temperature, or a power output, then take a rolling
        # average
        if any(
            (
                key in data_entry_name
                for key in [
                    "temperature",
                    "irradiance",
                    "efficiency",
                    "hot_water_load",
                    "electrical_load",
                ]
            )
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
        elif any((key in data_entry_name for key in {"load", "output"})):
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
        data_entry["absorber_temperature_gain"] = (
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
    x_series = [data_entry[TIME_KEY] for data_entry in model_data.values()]
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


def plot(  # pylint: disable=too-many-branches
    label: str,
    use_data_keys: bool,
    x_axis_key: str,
    x_label: str,
    y_label: str,
    model_data: Dict[Any, Any],
    *,
    disable_lines: bool,
    axes=None,
    bar_plot: bool = False,
    colour: str = None,
    hold=False,
    shape: str = "x",
) -> Optional[Any]:
    """
    Plots some model_data based on input parameters.

    :param label:
        The name of the model_data label to plot. For example, this could be
        "pv_temperature", in which case, the pv temperature would be plotted at the time
        steps specified in the model.

    :param use_data_keys:
        If specified, the keys of the data will be used. Otherwise, the value provided
        via the x-axis key will be used.

    :param x_axis_key:
        Key used for the x-axis data set. This defaults to the time of day.

    :param x_label:
        The x-axis label.

    :param y_label:
        The label to assign to the y axis when plotting.

    :param resolution:
        The resolution of the model that was run. This is measured in minutes.

    :param model_data:
        The model_data, loaded from the model's output file.

    :param disable_lines:
        If specified, lines will be disabled from the output.

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

    # Use the data keys themselves if appropriate, otherwise use the x-label provided.
    if use_data_keys:
        x_model_data = [float(entry) for entry in model_data.keys()]
    else:
        x_model_data = [entry[x_axis_key] for entry in model_data.values()]
    y_model_data = [entry[label] for entry in model_data.values()]

    # Sort the data series based on increasing x values if the data is not time.
    if x_axis_key != TIME_KEY:
        x_model_data, y_model_data = (
            list(entry) for entry in zip(*sorted(zip(x_model_data, y_model_data)))
        )

    # If we are not holding the graph, then clear the model_data.
    if not hold:
        plt.clf()
        plt.close("all")

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
            if not disable_lines:
                if colour is None:
                    (line,) = axes.plot(
                        x_model_data, y_model_data, label=label, marker=shape
                    )
                else:
                    (line,) = axes.plot(
                        x_model_data,
                        y_model_data,
                        label=label,
                        marker=shape,
                        color=colour,
                    )

    # Set the labels for the axes.
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return line if not disable_lines else None


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
    use_data_keys_as_x_axis: bool = False,
    x_axis_thing_to_plot: str = TIME_KEY,
    x_axis_label: str = "Time of day",
    second_axis_things_to_plot: Optional[List[str]] = None,
    second_axis_label: Optional[str] = None,
    second_axis_y_limits: Optional[Tuple[int, int]] = None,
    annotate_maximum: bool = False,
    bar_plot: bool = False,
    disable_lines: bool = False,
    plot_title: Optional[str] = None,
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

    :param use_data_keys_as_x_axis:
        If set to `True`, then the keys of the data set will be used for the plotting
        rather than the data set specified via the x-axis label.

    :param x_axis_thing_to_plot:
        The variable name to plot on the x-axis.

    :param x_axis_label:
        The label to assign to the x-axis.

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

    :param disable_lines:
        If specified, lines will be disabled from the plotting.

    :param plot_title:
        If specified, a title is addded to the plot.

    """

    _, ax1 = plt.subplots()

    lines = [
        plot(
            entry,
            use_data_keys_as_x_axis,
            x_axis_thing_to_plot,
            x_axis_label,
            first_axis_label,
            model_data,
            hold=True,
            axes=ax1,
            shape=first_axis_shape,
            bar_plot=bar_plot,
            disable_lines=disable_lines,
        )
        for entry in first_axis_things_to_plot
    ]

    # Adjust the x-tick separation if appropriate.
    locs, _ = plt.xticks()
    if len(locs) > 2 * X_TICK_SEPARATION:
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

    # Plot a title if specified.
    if plot_title is not None:
        plt.title(plot_title)

    # Save the figure and return if only one axis is plotted.
    if second_axis_things_to_plot is None:
        if lines is not None:
            plt.legend(lines, first_axis_things_to_plot)  # , loc="upper left")
        else:
            ax1.legend(first_axis_things_to_plot)
        save_figure(figure_name)
        return

    # Second-axis plotting.
    ax2 = ax1.twinx()

    lines.extend(
        [
            plot(
                entry,
                use_data_keys_as_x_axis,
                x_axis_thing_to_plot,
                x_axis_label,
                second_axis_label,
                model_data,
                hold=True,
                axes=ax2,
                shape=".",
                bar_plot=bar_plot,
                disable_lines=disable_lines,
            )
            for entry in second_axis_things_to_plot
        ]
    )

    if lines is not None:
        plt.legend(lines, first_axis_things_to_plot + second_axis_things_to_plot)
    else:
        ax2.legend(first_axis_things_to_plot + second_axis_things_to_plot)

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
    plot_title: str,
    entry_number: Optional[int] = None,
    hold: bool = False,
    hour: Optional[int] = None,
    minute: Optional[int] = None,
    x_axis_label: str = "Y segment index",
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

    :param plot_title:
        The title to use for the plot.

    :param entry_number:
        If provided, this is used to compute the entry to plot. Otherwise, hour and
        minute are used.

    :param hold:
        Whether to hold the plot.

    :param hour:
        The hour at which to plot the two-dimensional temperature profile.

    :param minute:
        The minute at which to plot the two-dimensional temperature profile.

    :param x_axis_label:
        The label to use for the x-axis if the plot is 1D.

    """

    # Determine the data index number based on the time.
    if entry_number is None:
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
    array_shape = (len(set(y_series)), len(set(x_series)))

    # If the data is only 1D, then plot a standard 1D profile through the absorber.
    if 1 in array_shape:
        # If we are not holding the graph, then clear the model_data.
        if not hold:
            plt.clf()
        plt.scatter(y_series, z_series)
        lines = plt.plot(y_series, z_series)
        # Set the labels for the axes.
        plt.xlabel(x_axis_label)
        plt.ylabel(axis_label)
        # Add the legend.
        axis = plt.gca()
        if lines is not None:
            plt.legend(lines, [thing_to_plot])
        else:
            axis.legend([thing_to_plot])
        save_figure(figure_name)

        return

    z_array = numpy.zeros(array_shape)
    for index, value in enumerate(z_series):
        z_array[len(set(y_series)) - (y_series[index] + 1), x_series[index]] = value

    # Plot the figure.
    fig3D = plt.figure()
    surface = plt.imshow(
        # axes3D,
        z_array,
        cmap=cm.coolwarm,  # pylint: disable=no-member
        # linewidth=0,
        # antialiased=False,
    )
    plt.title(plot_title)
    plt.xlabel("Segment x index")
    plt.ylabel("Segment y index")

    # Add axes and colour scale.
    fig3D.colorbar(surface, shrink=0.5, aspect=5, label=axis_label)

    save_figure(figure_name)

    x_array = numpy.reshape(x_series, array_shape)
    y_array = numpy.reshape(y_series, array_shape)
    z_array = numpy.reshape(z_series, array_shape)

    # Plot the figure.
    fig3D = plt.figure()
    axes3D = fig3D.gca(projection="3d")
    surface = plt3D.scatter(
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

    save_figure(f"{figure_name}_3d")
    if not hold:
        plt.close("all")


def analyse_dynamic_data(data: Dict[Any, Any], logger: Logger) -> None:
    """
    Carry out analysis on a set of dynamic data.

    :param data:
        The data to analyse.

    :param logger:
        The logger to use for the analysis run.

    """

    logger.info("Beginning analysis of dynamic data set.")

    # * Reduce the resolution of the data.
    data = _reduce_data(data, GRAPH_DETAIL, logger)
    logger.info(
        "Data successfully reduced to %s graph detail level.", GRAPH_DETAIL.name
    )

    # * Create new data values where needed.
    data = _post_process_data(data)
    logger.info("Post-processing of data complete.")

    # Plot All Temperatures
    plot_figure(
        "all_temperatures",
        data,
        first_axis_things_to_plot=[
            "ambient_temperature",
            "bulk_water_temperature",
            "absorber_temperature",
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
            "absorber_temperature",
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
            "absorber_temperature",
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
            "absorber_temperature_gain",
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
        plot_title="Glass layer temperature profile at 00:00",
    )

    plot_two_dimensional_figure(
        "glass_temperature_0600",
        logger,
        data,
        "layer_temperature_map_glass",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
        plot_title="Glass layer temperature profile at 06:00",
    )

    plot_two_dimensional_figure(
        "glass_temperature_1200",
        logger,
        data,
        "layer_temperature_map_glass",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
        plot_title="Glass layer temperature profile at 12:00",
    )

    plot_two_dimensional_figure(
        "glass_temperature_1800",
        logger,
        data,
        "layer_temperature_map_glass",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
        plot_title="Glass layer temperature profile at 18:00",
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
        plot_title="PV layer temperature profile at 00:00",
    )

    plot_two_dimensional_figure(
        "pv_temperature_0600",
        logger,
        data,
        "layer_temperature_map_pv",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
        plot_title="PV layer temperature profile at 06:00",
    )

    plot_two_dimensional_figure(
        "pv_temperature_1200",
        logger,
        data,
        "layer_temperature_map_pv",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
        plot_title="PV layer temperature profile at 12:00",
    )

    plot_two_dimensional_figure(
        "pv_temperature_1800",
        logger,
        data,
        "layer_temperature_map_pv",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
        plot_title="PV layer temperature profile at 18:00",
    )

    # Plot absorber layer temperatures at midnight, 6 am, noon, and 6 pm.
    plot_two_dimensional_figure(
        "absorber_temperature_0000",
        logger,
        data,
        "layer_temperature_map_absorber",
        axis_label="Temperature / degC",
        hour=0,
        minute=0,
        plot_title="Absorber layer temperature profile at 00:00",
    )

    plot_two_dimensional_figure(
        "absorber_temperature_0600",
        logger,
        data,
        "layer_temperature_map_absorber",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
        plot_title="Absorber layer temperature profile at 06:00",
    )

    plot_two_dimensional_figure(
        "absorber_temperature_1200",
        logger,
        data,
        "layer_temperature_map_absorber",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
        plot_title="Absorber layer temperature profile at 12:00",
    )

    plot_two_dimensional_figure(
        "absorber_temperature_1800",
        logger,
        data,
        "layer_temperature_map_absorber",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
        plot_title="Absorber layer temperature profile at 18:00",
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
        plot_title="Pipe temperature profile at 00:00",
    )

    plot_two_dimensional_figure(
        "pipe_temperature_0600",
        logger,
        data,
        "layer_temperature_map_pipe",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
        plot_title="Pipe temperature profile at 06:00",
    )

    plot_two_dimensional_figure(
        "pipe_temperature_1200",
        logger,
        data,
        "layer_temperature_map_pipe",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
        plot_title="Pipe temperature profile at 12:00",
    )

    plot_two_dimensional_figure(
        "pipe_temperature_1800",
        logger,
        data,
        "layer_temperature_map_pipe",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
        plot_title="Pipe temperature profile at 18:00",
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
        plot_title="Bulk-water temperature profile at 00:00",
    )

    plot_two_dimensional_figure(
        "bulk_water_temperature_0600",
        logger,
        data,
        "layer_temperature_map_bulk_water",
        axis_label="Temperature / degC",
        hour=6,
        minute=0,
        plot_title="Bulk-water temperature profile at 06:00",
    )

    plot_two_dimensional_figure(
        "bulk_water_temperature_1200",
        logger,
        data,
        "layer_temperature_map_bulk_water",
        axis_label="Temperature / degC",
        hour=12,
        minute=0,
        plot_title="Bulk-water temperature profile at 12:00",
    )

    plot_two_dimensional_figure(
        "bulk_water_temperature_1800",
        logger,
        data,
        "layer_temperature_map_bulk_water",
        axis_label="Temperature / degC",
        hour=18,
        minute=0,
        plot_title="Bulk-water temperature profile at 18:00",
    )

    """  # pylint: disable=pointless-string-statement
    # * Plotting all tank-related temperatures
    plot_figure(
        "tank_temperature",
        data,
        [
            "absorber_temperature",
            "collector_output_temperature",
            "absorber_temperature_gain",
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
            "absorber_temperature",
            "collector_output_temperature",
            "absorber_temperature_gain",
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
            "absorber_temperature",
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
            "absorber_temperature",
            "ambient_temperature",
            "sky_temperature",
        ],
        "Temperature / degC",
    )

    # * Plotting thermal-absorber-only temperatures
    plot_figure(
        "isolated_thermal_absorber",
        data,
        [
            "absorber_temperature",
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

    # * Plotting the absorber input, output, gain, and temperature.
    plot_figure(
        "absorber_temperatures",
        data,
        [
            "absorber_temperature",
            "collector_output_temperature",
            "collector_input_temperature",
            "absorber_temperature_gain",
            "tank_temperature",
        ],
        "Temperature / K",
    )

    # * Plotting the tank temperature, absorber temperature, and heat inputted into the
    # * tank.
    plot_figure(
        "tank_heat_gain_profile",
        data,
        first_axis_things_to_plot=["tank_temperature", "absorber_temperature"],
        first_axis_label="Temperature / deg C",
        first_axis_y_limits=(0, 100),
        second_axis_things_to_plot=["tank_heat_addition"],
        second_axis_label="Tank Heat Input / Watts",
    )

    # * Plotting the tank temperature, absorber temperature, and heat inputted into the
    # * tank.
    """  # pylint: disable=pointless-string-statement


def analyse_steady_state_data(data: Dict[Any, Any], logger: Logger) -> None:
    """
    Carry out analysis on a set of steady-state data.

    :param data:
        The data to analyse.

    :param logger:
        The logger to use for the analysis run.

    """

    logger.info("Beginning steady-state analysis.")

    print(f"{int(len(data.keys()) * 5 + 2)} figures will be plotted.")
    logger.info("%s figures will be plotted.", int(len(data.keys()) * 5 + 2))

    for temperature in data.keys():
        temperature_string = str(round(float(temperature), 2)).replace(".", "_")

        # Glass Temperatures
        logger.info("Plotting 3D glass profile at %s degC.", temperature_string)
        plot_two_dimensional_figure(
            "steady_state_glass_layer_{}degC_input".format(temperature_string),
            logger,
            data,
            axis_label="Temperature / deg C",
            entry_number=temperature,
            plot_title="Glass layer temperature with {} K input HTF".format(
                round(float(temperature), 2)
            ),
            thing_to_plot="layer_temperature_map_glass",
        )

        # PV Temperatures
        logger.info("Plotting 3D PV profile at %s degC.", temperature_string)
        plot_two_dimensional_figure(
            "steady_state_pv_layer_{}degC_input".format(temperature_string),
            logger,
            data,
            axis_label="Temperature / deg C",
            entry_number=temperature,
            plot_title="PV layer temperature with {} K input HTF".format(
                round(float(temperature), 2)
            ),
            thing_to_plot="layer_temperature_map_pv",
        )

        # Collector Temperatures
        logger.info("Plotting 3D absorber profile at %s degC.", temperature_string)
        plot_two_dimensional_figure(
            "steady_state_absorber_layer_{}degC_input".format(temperature_string),
            logger,
            data,
            axis_label="Temperature / deg C",
            entry_number=temperature,
            plot_title="Collector layer temperature with {} K input HTF".format(
                round(float(temperature), 2)
            ),
            thing_to_plot="layer_temperature_map_absorber",
        )

        # Pipe Temperatures
        logger.info(
            "Plotting 3D pipe profile at %s degC. NOTE: The profile will appear 2D if "
            "only one pipe is present.",
            temperature_string,
        )
        plot_two_dimensional_figure(
            "steady_state_pipe_{}degC_input".format(temperature_string),
            logger,
            data,
            axis_label="Pipe temperature / deg C",
            entry_number=temperature,
            plot_title="Pipe temperature with {} K input HTF".format(
                round(float(temperature), 2)
            ),
            thing_to_plot="layer_temperature_map_pipe",
        )

        # Bulk-water Temperatures
        logger.info(
            "Plotting 3D bulk-water profile at %s degC. NOTE: The profile will appear "
            "2D if only one pipe is present.",
            temperature_string,
        )
        plot_two_dimensional_figure(
            "steady_state_bulk_water_{}degC_input".format(temperature_string),
            logger,
            data,
            axis_label="Bulk-water temperature / deg C",
            entry_number=temperature,
            plot_title="Bulk-water temperature with {} K input HTF".format(
                round(float(temperature), 2)
            ),
            thing_to_plot="layer_temperature_map_bulk_water",
        )

    # Thermal efficiency plot.
    logger.info("Plotting thermal efficiency against the reduced temperature.")
    plot_figure(
        "thermal_efficiency_against_reduced_temperature",
        data,
        first_axis_things_to_plot=["thermal_efficiency"],
        first_axis_label="Thermal efficiency",
        x_axis_label="Reduced temperature / K m^2 / W",
        x_axis_thing_to_plot="reduced_collector_temperature",
        plot_title="Thermal efficiency against reduced temperature",
        disable_lines=True,
    )

    # Collector temperature gain plot.
    logger.info(
        "Plotting collector temperature gain against the input HTF temperature."
    )
    plot_figure(
        "collector_tempreature_gain_against_input_temperature",
        data,
        first_axis_things_to_plot=["collector_temperature_gain"],
        first_axis_label="Collector temperature gain / K",
        x_axis_label="Collector input temperature / degC",
        use_data_keys_as_x_axis=True,
        plot_title="Collector temperature gain against input temperature",
        disable_lines=True,
    )


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

    # * Determine whether the data is dynamic or steady-state.
    try:
        data_type = data.pop("data_type")
    except KeyError:
        logger.error(
            "Analysis data without an explicit data type is depreciated. Dynamic assumed."
        )
        data_type = DYNAMIC_DATA_TYPE

    # * Carry out analysis appropriate to the data type specified.
    if data_type == DYNAMIC_DATA_TYPE:
        analyse_dynamic_data(data, logger)
    elif data_type == STEADY_STATE_DATA_TYPE:
        analyse_steady_state_data(data, logger)
    else:
        logger.error("Data type was neither 'dynamic' nor 'steady_state'. Exiting...")
        sys.exit(1)

    logger.info("Analysis complete - all figures saved successfully.")

    if show_output:
        plt.show()


if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])
    analyse(parsed_args.data_file_name, parsed_args.show_output)
