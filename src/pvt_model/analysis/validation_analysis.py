#!/usr/bin/python3.7
# type: ignore
########################################################################################
# validation_analysis.py - The glazing analysis component for the model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################
"""
Used for analysis of various differences between outputs from model runs.

NOTE: The mypy type checker is instructed to ignore this component. This is done due to
the lower standards applied to the analysis code, and the failure of mypy to correctly
type-check the external matplotlib.pyplot module.

"""

import argparse
import collections
import os
import sys

from logging import Logger
from typing import Any, List, Dict, Optional, Set

import re

import numpy
import yaml

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
# Used to distinguish steady-state data sets.
STEADY_STATE_DATA_TYPE = "steady_state"
# Name of the directory containing validation data.
VALIDATION_DATA_DIRECTORY = "validation_data"
# How detailed the graph should be
GRAPH_DETAIL: GraphDetail = GraphDetail.lowest
# How many values there should be between each tick on the x-axis
# X_TICK_SEPARATION: int = int(8 * GRAPH_DETAIL.value / 48)
X_TICK_SEPARATION: int = 8
# Which days of data to include
DAYS_TO_INCLUDE: List[bool] = [False, True]


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


def _validation_figure(
    data: Dict[Any, Dict[str, Any]],
    logger: Logger,
    prefix: str,
    regex: re.Pattern,
    variable_name: str,
    *,
    validation_filename: Optional[str] = None,
) -> None:
    """
    Carries out analysis on a validation figure.

    Based on the parameters passed in, a sub-set of files is selected for, Ilaria's
    data is parsed and plotted if relevant, and a figure is saved.

    :param data:
        A `dict` mapping filenames to raw JSON data extracted from the files.

    :param logger:
        The logger to use for the run.

    :param prefrix:
        A `str` to prepend to all filenames so that the output figures can be uniquely
        identified.

    :param regex:
        A compiled :class:`re.Pattern` to use for matching and variable parsing from
        filenames.
        NOTE: The regex should contain:
        - either: the groups "first_digit" and "second_digit" which, together, give the
          value of the variable that identifies the data set,
        - or: the group "variable_value" which gives the value of the variable that
          identifies the data set.

    :param variable_name:
        The name of the variable being experimented with. This is used for labelling
        data sets and trendlines.

    :param validation_filename:
        If specified, this gives the name of the validation data file from which
        validation data should be plotted. If `None`, then no validation data is parsed
        and plotted.

    """

    # Plot the glass-emissivity effect on a single-glazed Ilaria.
    reduced_data: Dict[str, Any] = collections.defaultdict(dict)
    thermal_efficiency_labels: Set[str] = set()
    electrical_efficiency_labels: Set[str] = set()
    for key, sub_dict in data.items():
        match = re.match(regex, key)
        if match is None:
            continue
        try:
            variable_value = float(
                f"{match.group('first_digit')}.{match.group('second_digit')}"
            )
        except IndexError:
            variable_value = match.group("variable_value")
        # Only plot significant portions covered.
        for sub_key, value in sub_dict.items():
            # Generate a unique indentifier string.
            file_identifier = f"{variable_name}={variable_value}"
            thermal_efficiency_identifier = f"{file_identifier} (therm.)"
            electrical_efficiency_identifier = f"{file_identifier} (elec.)"
            # Store the specific thermal efficiency.
            reduced_data[sub_key][thermal_efficiency_identifier] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(thermal_efficiency_identifier)
            # Store the specific electrical efficiency.
            reduced_data[sub_key][electrical_efficiency_identifier] = value[
                "electrical_efficiency"
            ]
            electrical_efficiency_labels.add(electrical_efficiency_identifier)
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    # Thermal efficiency plot.
    logger.info("Plotting thermal efficiency against the reduced temperature.")
    _, ax1 = plt.subplots()

    # Plot Ilaria's data to validate against if relevant.
    if validation_filename is not None:
        # Parse the thermal-efficiency data.
        with open(
            os.path.join(VALIDATION_DATA_DIRECTORY, validation_filename),
            "r",
        ) as f:  #
            validation_data = yaml.safe_load(f)

        validation_data_dict = {
            entry["label"]: entry["thermal_efficiency"] for entry in validation_data
        }

        x_data_series = numpy.linspace(-0.005, 0.035, 100)
        y_data_sets = dict()
        for data_entry_name in ["single_glazed", "double_glazed", "unglazed"]:
            data_entry = validation_data_dict[data_entry_name]
            y_data_sets[data_entry_name] = (
                data_entry["x_squared"] * x_data_series ** 2
                + data_entry["linear_x"] * x_data_series
                + data_entry["y_intercept"]
            )

        for data_entry_name in sorted(["double_glazed", "single_glazed", "unglazed"]):
            # Plot the experimental data.
            ax1.plot(
                x_data_series,
                y_data_sets[data_entry_name],
            )

    ax1.set_prop_cycle(None)

    # Thermal efficiency plot.
    logger.info("Plotting thermal efficiency against the reduced temperature.")
    plot_figure(
        f"{prefix}_thermal_efficiency_against_reduced_temperature",
        reduced_data,
        first_axis_things_to_plot=thermal_efficiency_labels,
        first_axis_label="Thermal efficiency",
        x_axis_label="Reduced temperature / K m^2 / W",
        x_axis_thing_to_plot="reduced_collector_temperature",
        plot_title="Thermal efficiency against reduced temperature",
        disable_lines=True,
        plot_trendline=True,
        override_axis=ax1,
    )

    # Plot the electrical efficiency against the reduced temperature.
    logger.info("Plotting electrical efficiency against the reduced temperature.")

    plot_figure(
        f"{prefix}_electrical_efficiency_against_reduced_temperature",
        reduced_data,
        first_axis_things_to_plot=electrical_efficiency_labels,
        first_axis_label="Electrical efficiency",
        x_axis_label="Reduced temperature / K m^2 / W",
        x_axis_thing_to_plot="reduced_collector_temperature",
        plot_title="Electrical efficiency against reduced temperature",
        disable_lines=True,
        plot_trendline=True,
    )


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

    logger.info("Beginning autotherm glass-transmissivity analysis.")
    autotherm_glass_transmissivity_regex = re.compile(
        r"autotherm[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_glass_transmissivity.*"
    )
    _validation_figure(
        data,
        logger,
        "autotherm_glass_transmissivity",
        autotherm_glass_transmissivity_regex,
        "glass_transmissivity",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning Ilaria glass-transmissivity analysis.")
    ilaria_glass_transmissivity_regex = re.compile(
        r"ilaria[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_glass_transmissivity.*"
    )
    _validation_figure(
        data,
        logger,
        "ilaria_glass_transmissivity",
        ilaria_glass_transmissivity_regex,
        "glass_transmissivity",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning autotherm glass-emissivity analysis.")
    autotherm_glass_emissivity_regex = re.compile(
        r"autotherm[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_glass_emissivity.*"
    )
    _validation_figure(
        data,
        logger,
        "autotherm_glass_emissivity",
        autotherm_glass_emissivity_regex,
        "glass_emissivity",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning Ilaria glass-emissivity analysis.")
    ilaria_glass_emissivity_regex = re.compile(
        r"ilaria[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_glass_emissivity.*"
    )
    _validation_figure(
        data,
        logger,
        "ilaria_glass_emissivity",
        ilaria_glass_emissivity_regex,
        "glass_emissivity",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning autotherm pv-efficiency analysis.")
    autotherm_pv_efficiency_regex = re.compile(
        r"autotherm[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_pv_efficiency.*"
    )
    _validation_figure(
        data,
        logger,
        "autotherm_pv_efficiency",
        autotherm_pv_efficiency_regex,
        "pv_efficiency",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning Ilaria pv-efficiency analysis.")
    ilaria_pv_efficiency_regex = re.compile(
        r"ilaria[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_pv_efficiency.*"
    )
    _validation_figure(
        data,
        logger,
        "ilaria_pv_efficiency",
        ilaria_pv_efficiency_regex,
        "pv_efficiency",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning autotherm pv-emissivity analysis.")
    autotherm_pv_emissivity_regex = re.compile(
        r"autotherm[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_pv_emissivity.*"
    )
    _validation_figure(
        data,
        logger,
        "autotherm_pv_emissivity",
        autotherm_pv_emissivity_regex,
        "pv_emissivity",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning Ilaria pv-emissivity analysis.")
    ilaria_pv_emissivity_regex = re.compile(
        r"ilaria[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_pv_emissivity.*"
    )
    _validation_figure(
        data,
        logger,
        "ilaria_pv_emissivity",
        ilaria_pv_emissivity_regex,
        "pv_emissivity",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning autotherm thermal coefficient analysis.")
    autotherm_thermal_coefficient_regex = re.compile(
        r"autotherm[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_thermal_coefficient.*"
    )
    _validation_figure(
        data,
        logger,
        "autotherm_thermal_coefficient",
        autotherm_thermal_coefficient_regex,
        "thermal_coefficient",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning Ilaria thermal coefficient analysis.")
    ilaria_thermal_coefficient_regex = re.compile(
        r"ilaria[^\d]*(?P<first_digit>[\d]*)_(?P<second_digit>[\d]*)_thermal_coefficient.*"
    )
    _validation_figure(
        data,
        logger,
        "ilaria_thermal_coefficient",
        ilaria_thermal_coefficient_regex,
        "thermal_coefficient",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning autotherm glazing analysis.")
    autotherm_glazing_regex = re.compile(r"autotherm_(?P<variable_value>.*)_pc_1_0_.*")
    _validation_figure(
        data,
        logger,
        "autotherm_glazing",
        autotherm_glazing_regex,
        "glazing",
    )
    logger.info("Done.")
    plt.close()

    logger.info("Beginning Ilaria glazing analysis.")
    ilaria_glazing_regex = re.compile(r"ilaria_(?P<variable_value>.*)_pc_1_0_.*")
    _validation_figure(
        data,
        logger,
        "ilaria_glazing",
        ilaria_glazing_regex,
        "glazing",
        validation_filename="glazing_comparison.yaml",
    )
    logger.info("Done.")
    plt.close()

    logger.info("All validation analysis complete.")

    logger.info("Beginning Ilaria theoretical glazing analysis.")
    ilaria_glazing_regex = re.compile(
        r"ilarias_theoretical_panel_(?P<variable_value>.*).json"
    )
    _validation_figure(
        data,
        logger,
        "ilaria_glazing",
        ilaria_glazing_regex,
        "glazing",
        validation_filename="glazing_comparison.yaml",
    )
    logger.info("Done.")
    plt.close()

    logger.info("All validation analysis complete.")


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
        data[data_file_name] = load_model_data(
            os.path.join(data_file_directory, data_file_name)
        )

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
        if "no_pv" in data_file_name:
            logger.info("Invalid data was removed.")
            data.pop(data_file_name)
            continue

    analyse_decoupled_steady_state_data(data, logger)

    logger.info("Analysis complete - all figures saved successfully.")

    if show_output:
        plt.show()


if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])
    analyse(parsed_args.data_file_directory, parsed_args.show_output)
