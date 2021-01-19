#!/usr/bin/python3
"""
Does some analysis.

"""

import logging
import os
import pdb

from typing import Any, List, Dict, Optional, Tuple

import json
import re
import yaml

from matplotlib import pyplot as plt

from __utils__ import ProgrammerJudgementFault, GraphDetail, get_logger

# The directory in which old figures are saved
OLD_FIGURES_DIRECTORY: str = "old_figures"
# How detailed the graph should be
GRAPH_DETAIL: GraphDetail = GraphDetail.low
# How many values there should be between each tick on the x-axis
X_TICK_SEPARATION: int = int(8 * GRAPH_DETAIL.value / 48)
# Which days of data to include
DAYS_TO_INCLUDE: List[bool] = [False, True]
# The name of the data file to use.
DATA_FILE_NAME = "data_output_july_days_new_method_average_irradiance.json"


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
    if graph_detail == GraphDetail.lowest:
        # if int(num_data_points / 48) != num_data_points / 48:
        #     raise ProgrammerJudgementFault(
        #         "The number of data points recorded is not divisible by 48."
        #     )
        return int(num_data_points / GraphDetail.lowest.value)

    # * For "low", include one point every ten minutes
    if graph_detail == GraphDetail.low:
        # if int(num_data_points / (24 * 6)) != num_data_points / (24 * 6):
        #     raise ProgrammerJudgementFault(
        #         "The number of data points recorded is not divisible by {}.".format(
        #             str(24 * 6)
        #         )
        #     )
        return int(num_data_points / GraphDetail.low.value)

    # * For "medium", include one point every two minutes
    if graph_detail == GraphDetail.medium:
        # if int(num_data_points / (24 * 30)) != num_data_points / (24 * 30):
        #     raise ProgrammerJudgementFault(
        #         "The number of data points recorded is not divisible by {}.".format(
        #             str(24 * 30)
        #         )
        #     )
        return int(num_data_points / GraphDetail.medium.value)

    # * For "high", include one point every thirty seconds
    if graph_detail == GraphDetail.high:
        # if int(num_data_points / (24 * 60 * 2)) != num_data_points / (24 * 60 * 2):
        #     raise ProgrammerJudgementFault(
        #         "The number of data points recorded is not divisible by {}.".format(
        #             str(24 * 60 * 2)
        #         )
        #     )
        return int(num_data_points / GraphDetail.high.value)

    # * For highest, include all data points
    return 1


def _reduce_data(
    data: Dict[str, Dict[Any, Any]], graph_detail: GraphDetail
) -> Dict[str, Dict[Any, Any]]:
    """
    This processes the data, using sums to reduce the resolution so it can be plotted.

    :param data:
        The raw, JSON data, contained within a `dict`.

    :param graph_detail:
        The level of detail required in the graph.

    :return:
        The cropped/summed up data, returned at a lower resolution as specified by the
        graph detail.

    """

    # * First, only include the bits of data we want.
    # @@@ This only works for two days so far:
    # data = dict(list(data.items())[int(len(data) / 2) :])
    # data = {
    #     str(int(key) - 86400): value
    #     for key, value in data.items()
    # }

    data_points_per_graph_point: int = _resolution_from_graph_detail(
        graph_detail, len(data)
    )

    reduced_data: Dict[str : Dict[Any, Any]] = {
        index: dict() for index in range(int(len(data) / data_points_per_graph_point))
    }

    # Depending on the type of data entry, i.e., whether it is a temperature, load,
    # demand covered, or irradiance (or other), the way that it is processed will vary.
    for data_entry_name in data["0"].keys():
        # pdb.set_trace(header="Beginning of reduction loop.")
        # * If the entry is a date or time, just take the value
        if data_entry_name in ["date", "time"]:
            for index, _ in enumerate(reduced_data):
                reduced_data[index][data_entry_name] = data[
                    str(index * data_points_per_graph_point)
                ][data_entry_name]
            continue

        # * If the data entry is a temperature, or a power output, then take a rolling
        # average
        if any(
            [
                key in data_entry_name
                for key in ["temperature", "irradiance", "efficiency", "electrical"]
                if key != "dc_electrical"
            ]
        ):
            try:
                for outer_index, _ in enumerate(reduced_data):
                    reduced_data[outer_index][data_entry_name] = sum(
                        [
                            float(data[str(inner_index)][data_entry_name])
                            / data_points_per_graph_point
                            for inner_index in range(
                                int(data_points_per_graph_point * outer_index),
                                int(data_points_per_graph_point * (outer_index + 1)),
                            )
                        ]
                    )
                continue
            except TypeError as e:
                logger.error("A value was none. Setting to 'None': %s", str(e))
                for index, _ in enumerate(reduced_data):
                    reduced_data[index][data_entry_name] = None
                continue

        # * If the data entry is a load, then take a sum
        elif any([key in data_entry_name for key in ["load", "output"]]):
            # @@@
            # FIXME
            # * Here, the data is divided by 3600 to convert from Joules to Watt Hours.
            # * This only works provided that we are dealing with values in Joules...
            for outer_index, _ in enumerate(reduced_data):
                reduced_data[outer_index][data_entry_name] = (
                    sum(
                        [
                            float(data[str(inner_index)][data_entry_name])
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
            reduced_data[index][data_entry_name] = data[
                str(index * data_points_per_graph_point)
            ][data_entry_name]
        continue

    return reduced_data


def load_model_data(filename: str) -> Dict[Any, Any]:
    """
    Loads some model_data that was generated by the model.

    :param filename:
        The name of the model_data file to open.

    :return:
        The JSON model_data, loaded as a `dict`.

    """

    with open(filename, "r") as f:
        return json.load(f)


def plot(
    label: str,
    y_label: str,
    model_data: Dict[Any, Any],
    hold=False,
    axes=None,
    shape: str = "x",
    colour: str = None,
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

    :param hold:
        Whether to hold the screen between plots (True) or reset it (False).

    :param axes:
        If provided, a separate axis is used for plotting the model_data.

    :param shape:
        This sets the shape of the marker for `matplotlib.pyplot` to use when plotting
        the model_data.

    :param colour:
        The colour to use for the plot.

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

    # If we are not using axes, then the model_data can be straight plotted...
    if axes is None:
        plt.scatter(x_model_data, y_model_data, label=label, marker=shape)
        (line,) = plt.plot(x_model_data, y_model_data, label=label, marker=shape)

    # ... otherwise, the model_data needs to be plotted on just on axis.
    else:
        axes.scatter(x_model_data, y_model_data, label=label, marker=shape)
        if colour is None:
            (line,) = axes.plot(x_model_data, y_model_data, label=label, marker=shape)
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
    if os.path.isfile(f"figure_{figure_name}_1.jpg"):
        os.rename(
            f"figure_{figure_name}_1.jpg",
            os.path.join(OLD_FIGURES_DIRECTORY, f"figure_{figure_name}_1.jpg"),
        )

    # If it exists, move the current figure to _1
    if os.path.isfile(f"figure_{figure_name}.jpg"):
        os.rename(f"figure_{figure_name}.jpg", f"figure_{figure_name}_1.jpg")

    # Save the figure
    plt.savefig(f"figure_{figure_name}.jpg")


def plot_figure(
    figure_name: str,
    model_data: Dict[Any, Any],
    first_axis_things_to_plot: List[str],
    first_axis_label: str,
    *,
    first_axis_y_limits: Optional[Tuple[int, int]] = None,
    second_axis_things_to_plot: Optional[List[str]] = None,
    second_axis_label: Optional[str] = None,
    second_axis_y_limits: Optional[Tuple[int, int]] = None,
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

    :param second_axis_things_to_plot:
        The list of variable names (keys in the JSON model_data) to plot on the second axis.

    :param second_axis_label:
        The label to assign to the second y-axis.

    :param second_axis_y_limits:
        A `tuple` giving the lower and upper limits to set for the y axis for the second
        axis.

    """

    # Generate the necessary local variables needed for sub-plotting.
    _, ax1 = plt.subplots()

    lines = [
        plot(
            entry,
            first_axis_label,
            model_data,
            hold=True,
            axes=ax1,
        )
        for entry in first_axis_things_to_plot
    ]

    ax1.set_xticks(ax1.get_xticks()[::X_TICK_SEPARATION])

    # Set the y limits if appropriate
    if first_axis_y_limits is not None:
        plt.ylim(*first_axis_y_limits)

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
            )
            for entry in second_axis_things_to_plot
        ]
    )

    plt.legend(lines, first_axis_things_to_plot + second_axis_things_to_plot)

    ax2.set_xticks(ax2.get_xticks()[::X_TICK_SEPARATION])

    # Set the y limits if appropriate.
    if second_axis_y_limits is not None:
        plt.ylim(*second_axis_y_limits)

    save_figure(figure_name)


if __name__ == "__main__":

    # * Set up the logger
    logger = get_logger("pvt_analysis")

    # * Extract the data.
    data = load_model_data(DATA_FILE_NAME)

    # * Reduce the resolution of the data.
    data = _reduce_data(data, GRAPH_DETAIL)

    # f Plot Figure 4a: Electrical Demand
    plot_figure(
        "maria_4a_electrical_load",
        data,
        ["electrical_load"],
        "Dwelling Load Profile / W",
    )

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

    # * Plotting the solar irradiance and irradiance normal to the panel
    plot_figure(
        "solar_irradiance",
        data,
        ["solar_irradiance", "normal_irradiance"],
        "Solar Irradiance / Watts / meter squared",
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

    # * Plotting Maria's figure 8A - Electrical Power and Net Electrical Power
    plot_figure(
        "electrical_output_8A",
        data,
        ["gross_electrical_output", "net_electrical_output"],
        "Electrical Power Output / W",
    )

    # * Plotting Maria's figure 10 - Electrical Power
    plot_figure(
        "gross_electrical_output_10",
        data,
        ["gross_electrical_output"],
        "Electrical Power Output / W",
    )

    # * Plotting Maria's figure 8B - Thermal Power Supplied and Thermal Power Demanded
    plot_figure(
        "thermal_output_8B",
        data,
        ["thermal_load", "thermal_output"],
        "Thermal Energy Supplied / Wh",
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
    plot_figure(
        "ambient_temperature",
        data,
        first_axis_things_to_plot=["ambient_temperature", "sky_temperature"],
        first_axis_label="Temperature / deg C",
    )
