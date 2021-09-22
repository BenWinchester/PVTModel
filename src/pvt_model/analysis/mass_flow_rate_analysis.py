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
import collections
import os
import sys

from logging import Logger
from typing import Any, List, Dict, Optional, Tuple

import json
import re

import numpy

from matplotlib import pyplot as plt

try:
    from ..__utils__ import get_logger
    from ..pvt_system.constants import (  # pylint: disable=unused-import
        HEAT_CAPACITY_OF_WATER,
    )
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
# Used to distinguish dynamic data sets.
DYNAMIC_DATA_TYPE = "dynamic"
# The directory into which which should be saved
NEW_FIGURES_DIRECTORY: str = "figures"
# Used to identify the "time" data set.
TIME_KEY = "time"
# The directory in which old figures are saved and stored for long-term access
OLD_FIGURES_DIRECTORY: str = "old_figures"
# Used to specify the reduced temperature for comparing the thermal performance of the
# collectors.
REDUCED_TEMPERATURE_COMPARISON_POINT = 0.10
# Used to distinguish steady-state data sets.
STEADY_STATE_DATA_TYPE = "steady_state"
# Name of the steady-state data file.
STEADY_STATE_DATA_FILE_NAME = "autotherm.yaml"
# How detailed the graph should be
GRAPH_DETAIL: GraphDetail = GraphDetail.lowest
# How many values there should be between each tick on the x-axis
# X_TICK_SEPARATION: int = int(8 * GRAPH_DETAIL.value / 48)
X_TICK_SEPARATION: int = 8
# Which days of data to include
DAYS_TO_INCLUDE: List[bool] = [False, True]
# Portion-covered regex.
MASS_FLOW_RATE_REGEX = re.compile(
    r"(?P<panel_type>[^_]*)_(?P<no_pv>no_pv_|)"
    r"(?P<glazing>single_|double_|un)glazed_(?P<pv>[^\d]*|)"
    r"(?P<mass_flow_rate>\d*_\d)_litres_per_hour_(?P<x_resolution>\d*)_x_"
    r"(?P<y_resolution>\d*).*"
)
# A regex is needed to exclude results with no pv layer.
NO_PV_REGEX = re.compile(r".*no_pv.*")


def _calculate_value_at_zero_reduced_temperature(
    filedata: Dict[Any, Any], parameter: str
) -> float:
    """
    Calculates the value of the parameter passed in at a reduced temperature of zero.

    :param filedata:
        The raw data extracted from the data file.

    :param parameter:
        A `str` specifying the parameter that should be computed at a reduced
        temperature of zero.

    :return:
        The value of the parameter specified at a reduced temperature of zero.

    """

    x_series: List[float] = []
    y_series: List[float] = []
    for key, value in filedata.items():
        x_series.append(float(key))
        y_series.append(float(value[parameter]))

    trend = numpy.polyfit(x_series, y_series, 2)
    return (
        trend[2] * REDUCED_TEMPERATURE_COMPARISON_POINT ** 2
        + trend[1] * REDUCED_TEMPERATURE_COMPARISON_POINT
        + trend[2]
    )


def _calculate_zero_point_efficiencies(filedata: Dict[Any, Any]) -> Tuple[float, float]:
    """
    Calculates the various values at a reduced temperature of zero Kelvin.

    :param filedata:
        The raw data extracted from the data file.

    :return:
        A `tuple` containing:
        - the electrical efficiency,
        - and the thermal efficiency
        at a reduced temperature of zero.

    """

    return (
        _calculate_value_at_zero_reduced_temperature(filedata, "electrical_efficiency"),
        _calculate_value_at_zero_reduced_temperature(filedata, "thermal_efficiency"),
    )


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
    return data_to_post_process


def analyse_decoupled_steady_state_data(  # pylint: disable=too-many-branches
    data: Dict[Any, Dict[str, Any]], logger: Logger
) -> None:
    """
    Carry out analysis on a series of of steady-state data sets.

    :param data:
        The data to analyse. This should be a `dict` mapping the file name to the data
        extracted from the file.

    :param logger:
        The logger to use for the analysis run.

    """

    logger.info("Beginning steady-state analysis.")

    # Construct a reduced mapping
    autotherm_single_glazed_data: Dict[str, Any] = collections.defaultdict(dict)
    autotherm_no_pv_single_glazed_data: Dict[str, Any] = collections.defaultdict(dict)
    autotherm_double_glazed_data: Dict[str, Any] = collections.defaultdict(dict)
    autotherm_no_pv_double_glazed_data: Dict[str, Any] = collections.defaultdict(dict)
    ilaria_single_glazed_data: Dict[str, Any] = collections.defaultdict(dict)
    ilaria_no_pv_single_glazed_data: Dict[str, Any] = collections.defaultdict(dict)
    ilaria_double_glazed_data: Dict[str, Any] = collections.defaultdict(dict)
    ilaria_no_pv_double_glazed_data: Dict[str, Any] = collections.defaultdict(dict)
    for key, sub_dict in data.items():
        mass_flow_rate_match = re.match(MASS_FLOW_RATE_REGEX, key)
        if mass_flow_rate_match is None:
            continue

        mass_flow_rate_string: str = mass_flow_rate_match.group("mass_flow_rate")
        try:
            mass_flow_rate: float = int(
                mass_flow_rate_string.split("_")[0]
            ) + 0.1 * int(mass_flow_rate_string.split("_")[1])
        except ValueError:
            mass_flow_rate = int(mass_flow_rate_string.split("_")[1])

        electrical_efficiency, thermal_efficiency = _calculate_zero_point_efficiencies(
            sub_dict
        )

        # If there was no pv layer present in the simulation.
        if mass_flow_rate_match.group("no_pv") == "":
            if mass_flow_rate_match.group("panel_type") == "autotherm":
                if f"{mass_flow_rate_match.group('glazing')}glazed" == "single_glazed":
                    autotherm_single_glazed_data[mass_flow_rate][
                        "electrical_efficiency"
                    ] = electrical_efficiency
                    autotherm_single_glazed_data[mass_flow_rate][
                        "thermal_efficiency"
                    ] = thermal_efficiency
                elif (
                    f"{mass_flow_rate_match.group('glazing')}glazed" == "double_glazed"
                ):
                    autotherm_double_glazed_data[mass_flow_rate][
                        "electrical_efficiency"
                    ] = electrical_efficiency
                    autotherm_double_glazed_data[mass_flow_rate][
                        "thermal_efficiency"
                    ] = thermal_efficiency

            if mass_flow_rate_match.group("panel_type") == "ilaria":
                if f"{mass_flow_rate_match.group('glazing')}glazed" == "single_glazed":
                    ilaria_single_glazed_data[mass_flow_rate][
                        "electrical_efficiency"
                    ] = electrical_efficiency
                    ilaria_single_glazed_data[mass_flow_rate][
                        "thermal_efficiency"
                    ] = thermal_efficiency
                elif (
                    f"{mass_flow_rate_match.group('glazing')}glazed" == "double_glazed"
                ):
                    ilaria_double_glazed_data[mass_flow_rate][
                        "electrical_efficiency"
                    ] = electrical_efficiency
                    ilaria_double_glazed_data[mass_flow_rate][
                        "thermal_efficiency"
                    ] = thermal_efficiency
        else:
            if mass_flow_rate_match.group("panel_type") == "autotherm":
                if f"{mass_flow_rate_match.group('glazing')}glazed" == "single_glazed":
                    autotherm_no_pv_single_glazed_data[mass_flow_rate][
                        "electrical_efficiency"
                    ] = electrical_efficiency
                    autotherm_no_pv_single_glazed_data[mass_flow_rate][
                        "thermal_efficiency"
                    ] = thermal_efficiency
                elif (
                    f"{mass_flow_rate_match.group('glazing')}glazed" == "double_glazed"
                ):
                    autotherm_no_pv_double_glazed_data[mass_flow_rate][
                        "electrical_efficiency"
                    ] = electrical_efficiency
                    autotherm_no_pv_double_glazed_data[mass_flow_rate][
                        "thermal_efficiency"
                    ] = thermal_efficiency

            if mass_flow_rate_match.group("panel_type") == "ilaria":
                if f"{mass_flow_rate_match.group('glazing')}glazed" == "single_glazed":
                    ilaria_no_pv_single_glazed_data[mass_flow_rate][
                        "electrical_efficiency"
                    ] = electrical_efficiency
                    ilaria_no_pv_single_glazed_data[mass_flow_rate][
                        "thermal_efficiency"
                    ] = thermal_efficiency
                elif (
                    f"{mass_flow_rate_match.group('glazing')}glazed" == "double_glazed"
                ):
                    ilaria_no_pv_double_glazed_data[mass_flow_rate][
                        "electrical_efficiency"
                    ] = electrical_efficiency
                    ilaria_no_pv_double_glazed_data[mass_flow_rate][
                        "thermal_efficiency"
                    ] = thermal_efficiency

    # Plot the PV thermal efficiencies.
    plot_figure(
        "autotherm_single_glazed_thermal_efficiency_against_mass_flow_rate",
        autotherm_single_glazed_data,
        ["thermal_efficiency"],
        "Thermal efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "autotherm_double_glazed_thermal_efficiency_against_mass_flow_rate",
        autotherm_double_glazed_data,
        ["thermal_efficiency"],
        "Thermal efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "ilaria_single_glazed_thermal_efficiency_against_mass_flow_rate",
        ilaria_single_glazed_data,
        ["thermal_efficiency"],
        "Thermal efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "ilaria_double_glazed_thermal_efficiency_against_mass_flow_rate",
        ilaria_double_glazed_data,
        ["thermal_efficiency"],
        "Thermal efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    # Plot the no-PV thermal efficiencies.
    plot_figure(
        "autotherm_no_pv_single_glazed_thermal_efficiency_against_mass_flow_rate",
        autotherm_no_pv_single_glazed_data,
        ["thermal_efficiency"],
        "Thermal efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "autotherm_no_pv_double_glazed_thermal_efficiency_against_mass_flow_rate",
        autotherm_no_pv_double_glazed_data,
        ["thermal_efficiency"],
        "Thermal efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "ilaria_no_pv_single_glazed_thermal_efficiency_against_mass_flow_rate",
        ilaria_no_pv_single_glazed_data,
        ["thermal_efficiency"],
        "Thermal efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "ilaria_no_pv_double_glazed_thermal_efficiency_against_mass_flow_rate",
        ilaria_no_pv_double_glazed_data,
        ["thermal_efficiency"],
        "Thermal efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    # Plot the pv electrical efficiencies.
    plot_figure(
        "autotherm_single_glazed_electrical_efficiency_against_mass_flow_rate",
        autotherm_single_glazed_data,
        ["electrical_efficiency"],
        "Electrical efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "autotherm_double_glazed_electrical_efficiency_against_mass_flow_rate",
        autotherm_double_glazed_data,
        ["electrical_efficiency"],
        "Electrical efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "ilaria_single_glazed_electrical_efficiency_against_mass_flow_rate",
        ilaria_single_glazed_data,
        ["electrical_efficiency"],
        "Electrical efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "ilaria_double_glazed_electrical_efficiency_against_mass_flow_rate",
        ilaria_double_glazed_data,
        ["electrical_efficiency"],
        "Electrical efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    # Plot the no-PV electrical efficiencies.
    plot_figure(
        "autotherm_no_pv_single_glazed_electrical_efficiency_against_mass_flow_rate",
        autotherm_no_pv_single_glazed_data,
        ["electrical_efficiency"],
        "Electrical efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "autotherm_no_pv_double_glazed_electrical_efficiency_against_mass_flow_rate",
        autotherm_no_pv_double_glazed_data,
        ["electrical_efficiency"],
        "Electrical efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "ilaria_no_pv_single_glazed_electrical_efficiency_against_mass_flow_rate",
        ilaria_no_pv_single_glazed_data,
        ["electrical_efficiency"],
        "Electrical efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )

    plot_figure(
        "ilaria_no_pv_double_glazed_electrical_efficiency_against_mass_flow_rate",
        ilaria_no_pv_double_glazed_data,
        ["electrical_efficiency"],
        "Electrical efficiency",
        x_axis_label="Mass-flow rate / Litres per hour",
        use_data_keys_as_x_axis=True,
        disable_lines=True,
    )


def analyse(data_file_directory: str, show_output: Optional[bool] = False) -> None:
    """
    The main method for the analysis module.

    :param data_file_directory:
        The path to the directory containing data files to analyse.

    :param show_output:
        Whether to show the output files generated.

    """

    # * Set up the logger
    logger = get_logger(True, "pvt_analysis", True)

    # * Extract the data.
    data: Dict[str, Dict[str, Any]] = dict()
    for data_file_name in os.listdir(data_file_directory):
        try:
            data[data_file_name] = load_model_data(
                os.path.join(data_file_directory, data_file_name)
            )
        except json.JSONDecodeError:
            print(f"JSON decode error encountered. File name: {data_file_name}")
            raise

        # If the data type is not decoupled, steady-state data, then exit.
        data_type = data[data_file_name].pop("data_type")
        if data_type not in (
            STEADY_STATE_DATA_TYPE,
            f"{DECOUPLED_DATA_TYPE}_{STEADY_STATE_DATA_TYPE}",
        ):
            logger.info(
                "Data type was neither 'dynamic' nor 'steady_state'. Omitting..."
            )
            data.pop(data_file_name)
            continue

    analyse_decoupled_steady_state_data(data, logger)

    logger.info("Analysis complete - all figures saved successfully.")

    if show_output:
        plt.show()


if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])
    analyse(parsed_args.data_file_directory, parsed_args.show_output)
