#!/usr/bin/python3.7
# type: ignore
########################################################################################
# __utils__.py - The utility module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The utility module for the analysis component.

"""

import enum
import os

from logging import Logger
from typing import Any, List, Dict, Optional, Tuple

import re

import numpy

from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3D

try:
    from ..pvt_system_model.constants import (  # pylint: disable=unused-import
        HEAT_CAPACITY_OF_WATER,
    )
except ModuleNotFoundError:
    import logging

    logging.error(
        "Incorrect module import. Try running with `python3.7 -m pvt_model.analysis`"
    )
    raise


__all__ = ("GraphDetail", "plot", "plot_figure", "plot_two_dimensional_figure")

# Used to distinguish copled data sets.
COUPLED_DATA_TYPE = "coupled"
# Used to distinguished decoupled data sets.
DECOUPLED_DATA_TYPE = "decoupled"
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
# Name of the steady-state data file.
STEADY_STATE_DATA_FILE_NAME = "autotherm.yaml"
# How many values there should be between each tick on the x-axis
# X_TICK_SEPARATION: int = int(8 * GRAPH_DETAIL.value / 48)
X_TICK_SEPARATION: int = 8
# Which days of data to include
DAYS_TO_INCLUDE: List[bool] = [False, True]


class GraphDetail(enum.Enum):
    """
    The level of detail to go into when graphing.

    .. attribute:: highest
        The highest level of detail - all data points are plotted.

    .. attribute:: high
        A "high" level of detail, to be determined by the analysis script.

    .. attribute:; medium
        A "medium" level of detail, to be determined by the analysis script.

    .. attribute:: low
        A "low" level of detail, to be determined by the analysis script.

    .. attribute:: lowest
        The lowest level of detail, with points only every half an hour.

    """

    highest = 0
    high = 2880
    medium = 720
    low = 144
    lowest = 48


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
    hold: bool = False,
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


def plot_figure(  # pylint: disable=too-many-branches
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
    override_axis: Optional[Axes] = None,
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

    :param override_axis:
        If specified, this overrides the fetching of internal axes.

    :param plot_title:
        If specified, a title is addded to the plot.

    """

    if override_axis is None:
        _, ax1 = plt.subplots()
    else:
        ax1 = override_axis

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
    x_axis_label: str = "Y element index",
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
        aspect=array_shape[1] / array_shape[0],
    )
    plt.title(plot_title)
    plt.xlabel("Element x index")
    plt.ylabel("Element y index")

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
