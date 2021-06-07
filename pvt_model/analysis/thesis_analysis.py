#!/usr/bin/python3.7
# type: ignore
########################################################################################
# thesis_analysis.py - Carries out analysis on data available from Ilaria in her thesis.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################
"""
Carries out analysis to compare results produced by my model to Ilaria's thesis.

The glazing validation, analysed in `glazing_analysis.py`, has produced results that
differ from those obtained by Ilaria with her model. However, in her thesis, she has
experimental results that look more similar to those obtained by my model.

This module hence carries out analysis to plot my results against those obtained
experimentally by Ilaria for comparison and validation purposes.

"""

import argparse
import collections
import os
import sys

from logging import Logger
from typing import Any, Dict, Optional, Union

import re
import numpy
import yaml

from matplotlib import pyplot as plt

try:
    from ..__utils__ import get_logger
    from .__utils__ import (
        GraphDetail,
        load_model_data,
        plot_figure,
    )
except ModuleNotFoundError:
    import logging

    logging.error(
        "Incorrect module import. Try running with `python3.7 -m pvt_model.analysis`"
    )
    raise

__all__ = ("analyse",)

# Used to distinguish copled data sets.
COUPLED_DATA_TYPE = "coupled"
# Used to distinguished decoupled data sets.
DECOUPLED_DATA_TYPE = "decoupled"
# Used to identify the "time" data set.
TIME_KEY = "time"
# Used to distinguish steady-state data sets.
STEADY_STATE_DATA_TYPE = "steady_state"
# Name of the steady-state data file.
STEADY_STATE_DATA_FILE_NAME = "ilaria_glazing_validation_runs.yaml"
# Height of the y error bars.
Y_ERROR_BAR_HEIGHT = 0.1


def _parse_args(args) -> argparse.Namespace:
    """
    Parse the CLI args.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-file-directory",
        "-dfdir",
        help="Path to the directory containing data files to parse.",
    )
    parser.add_argument(
        "--show-output",
        "-so",
        action="store_true",
        default=False,
        help="Show the output figures generated.",
    )
    parser.add_argument(
        "--skip-2d-plots",
        "-skip",
        action="store_true",
        default=False,
        help="Skip plotting of 2D figures.",
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


def _plot_experimental_data() -> plt.Axes:
    """
    Plot the experimental data for the runs, both unglazed and single glazed.

    :return:
        The axes used for the plotting.

    """

    # Parse the thermal-efficiency data.
    with open(
        os.path.join("validation_data", STEADY_STATE_DATA_FILE_NAME),
        "r",
    ) as f:  #
        experimental_steady_state_data = yaml.safe_load(f)

    # Plot the experimental data.
    _, ax1 = plt.subplots()
    for glazing, colour in zip(["unglazed", "single_glazed"], ["blue", "orange"]):
        x_data = [
            entry["reduced_temperature"]
            for entry in experimental_steady_state_data[glazing]
        ]
        y_data = [
            entry["thermal_efficiency"]
            for entry in experimental_steady_state_data[glazing]
        ]
        error_bar_data = [
            entry["error_bar"] for entry in experimental_steady_state_data[glazing]
        ]
        ax1.scatter(
            x_data,
            y_data,
            # yerr=Y_ERROR_BAR_HEIGHT,
            marker="s",
            # ls="none",
        )
        plt.errorbar(
            x_data,
            y_data,
            yerr=error_bar_data,
            ls="none",
            color=colour,
        )
        fit = numpy.polyfit(x_data, y_data, 2)
        x_series = numpy.linspace(min(x_data), max(x_data), 100)
        y_series = [fit[0] * entry ** 2 + fit[1] * entry + fit[2] for entry in x_series]
        plt.plot(x_series, y_series)

    return ax1


def analyse_decoupled_steady_state_data(
    data: Dict[Any, Any], logger: Logger, skip_2d_plots: bool
) -> None:
    """
    Carry out analysis on a set of steady-state data.

    :param data:
        The data to analyse.

    :param logger:
        The logger to use for the analysis run.

    :param skip_2d_plots:
        Whether to skip the 2D plots (True) or plot them (False).

    """

    logger.info("Beginning steady-state analysis.")

    if skip_2d_plots:
        print(f"{int(len(data.keys()) * 2 + 2)} figures will be plotted.")
        logger.info("%s figures will be plotted.", int(len(data.keys()) * 2 + 2))
    else:
        print(f"{int(len(data.keys()) * 5 + 2)} figures will be plotted.")
        logger.info("%s figures will be plotted.", int(len(data.keys()) * 5 + 2))

    # Post-process the data to a plottable format.
    single_glazed_data = collections.defaultdict(dict)
    unglazed_data = collections.defaultdict(dict)
    single_glazed_thermal_efficiency_labels = set()
    unglazed_thermal_efficiency_labels = set()
    ilaria_glazing_regex = re.compile(r"ilaria_(?P<glazing>.*)_steady_state_runs.*")
    for key, sub_dict in data.items():
        ilaria_glazing_match = re.match(ilaria_glazing_regex, key)
        if ilaria_glazing_match is None:
            continue
        glazing = ilaria_glazing_match.group("glazing")
        if glazing == "single_glazed":
            for sub_key, value in sub_dict.items():
                single_glazed_data[sub_key][f"{glazing} thermal efficiency"] = value[
                    "thermal_efficiency"
                ]
                single_glazed_thermal_efficiency_labels.add(
                    f"{glazing} thermal efficiency"
                )
                # Store a copy of the reduced temperature
                single_glazed_data[sub_key]["reduced_collector_temperature"] = value[
                    "reduced_collector_temperature"
                ]
        else:
            for sub_key, value in sub_dict.items():
                unglazed_data[sub_key][f"{glazing} thermal efficiency"] = value[
                    "thermal_efficiency"
                ]
                unglazed_thermal_efficiency_labels.add(f"{glazing} thermal efficiency")
                # Store a copy of the reduced temperature
                unglazed_data[sub_key]["reduced_collector_temperature"] = value[
                    "reduced_collector_temperature"
                ]

    # Thermal efficiency plot.
    logger.info("Plotting thermal efficiency against the reduced temperature.")

    # Plot the experimental data.
    ax1 = _plot_experimental_data()

    # Add the model data.
    plot_figure(
        "thesis_glazing_analysis_thermal_efficiency_against_reduced_temperature",
        single_glazed_data,
        first_axis_things_to_plot=single_glazed_thermal_efficiency_labels,
        first_axis_label="Thermal efficiency",
        x_axis_label="Reduced temperature / K m^2 / W",
        x_axis_thing_to_plot="reduced_collector_temperature",
        plot_title="Thermal efficiency against reduced temperature",
        disable_lines=True,
        plot_trendline=True,
        override_axis=ax1,
        first_axis_y_limits=[-0.1, 0.6],
        transparent=True,
    )
    plot_figure(
        "thesis_glazing_analysis_thermal_efficiency_against_reduced_temperature",
        unglazed_data,
        first_axis_things_to_plot=unglazed_thermal_efficiency_labels,
        first_axis_label="Thermal efficiency",
        x_axis_label="Reduced temperature / K m^2 / W",
        x_axis_thing_to_plot="reduced_collector_temperature",
        plot_title="Thermal efficiency against reduced temperature",
        disable_lines=True,
        plot_trendline=True,
        override_axis=ax1,
        first_axis_y_limits=[-0.1, 0.6],
        transparent=True,
    )


def analyse(
    data_file_directory: str,
    show_output: Optional[bool] = False,
    skip_2d_plots: Optional[bool] = False,
) -> None:
    """
    The main method for the analysis module.

    :param data_file_directory:
        The path to the directory containing data files to analyse.

    :param show_output:
        Whether to show the output files generated.

    :param skip_2d_plots:
        Whether to skip the 2D plots (True) of include them (False).

    """

    # * Set up the logger
    logger = get_logger(True, "pvt_thesis_analysis", True)

    # * Extract the data.
    data: Dict[str, Dict[str, Any]] = dict()
    for data_file_name in os.listdir(data_file_directory):
        data[data_file_name] = load_model_data(
            os.path.join(data_file_directory, data_file_name)
        )

        # If the data type is not decoupled, steady-state data, then exit.
        data_type = data[data_file_name].pop("data_type")
        if data_type != f"{DECOUPLED_DATA_TYPE}_{STEADY_STATE_DATA_TYPE}":
            logger.info("Data type was not 'steady_state'. Omitting...")
            data.pop(data_file_name)
            continue
        if "no_pv" in data_file_name:
            logger.info("Invalid data was removed.")
            data.pop(data_file_name)
            continue

    analyse_decoupled_steady_state_data(data, logger, skip_2d_plots)

    logger.info("Analysis complete - all figures saved successfully.")

    if show_output:
        plt.show()


if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])
    analyse(
        parsed_args.data_file_directory,
        parsed_args.show_output,
        parsed_args.skip_2d_plots,
    )
