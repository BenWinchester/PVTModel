#!/usr/bin/python3.7
# type: ignore
########################################################################################
# glazing_analysis.py - The glazing analysis component for the model.
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
PORTION_COVERED_REGEX = re.compile(
    r".*pc_(?P<first_digit>[0-9]*)_(?P<second_digit>[0-9])_.*"
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


def _efficiency_plots(
    *,
    data: Dict[str, Any],
    electrical_efficiency_labels: List[str],
    logger: Logger,
    prefix: str,
    thermal_efficiency_labels: List[str],
) -> None:
    """
    Plot, based on the data provided, the thermal and electrical efficiencies.

    :param data:
        The data to be plotted of the type from each data file.

    :param electrical_efficiency_labels:
        A `list` containing the names assigned to the various electrical efficiency
        data sets stored in the dict entries.

    :param logger:
        The logger to use for the run.

    :param prefix:
        A prefix to put at the start of the file names.

    :param thermal_efficiency_labels:
        A `list` containing the names assigned to the various thermal efficiency data
        sets stored in the dict entries.

    """

    # Thermal efficiency plot.
    logger.info("Plotting thermal efficiency against the reduced temperature.")
    plot_figure(
        f"{prefix}_thermal_efficiency_against_reduced_temperature",
        data,
        first_axis_things_to_plot=thermal_efficiency_labels,
        first_axis_label="Thermal efficiency",
        x_axis_label="Reduced temperature / K m^2 / W",
        x_axis_thing_to_plot="reduced_collector_temperature",
        plot_title="Thermal efficiency against reduced temperature",
        disable_lines=True,
        plot_trendline=True,
    )

    # Collector temperature gain plot.
    # logger.info(
    #     "Plotting collector temperature gain against the input HTF temperature."
    # )

    # plot_figure(
    #     f"{prefix}_collector_tempreature_gain_against_input_temperature",
    #     data,
    #     first_axis_things_to_plot=["collector_temperature_gain"],
    #     first_axis_label="Collector temperature gain / K",
    #     x_axis_label="Collector input temperature / degC",
    #     use_data_keys_as_x_axis=True,
    #     plot_title="Collector temperature gain against input temperature",
    #     disable_lines=True,
    # )

    # Plot the electrical efficiency against the reduced temperature.
    logger.info("Plotting electrical efficiency against the reduced temperature.")

    plot_figure(
        f"{prefix}_electrical_efficiency_against_reduced_temperature",
        data,
        first_axis_things_to_plot=electrical_efficiency_labels,
        first_axis_label="Electrical efficiency",
        x_axis_label="Reduced temperature / K m^2 / W",
        x_axis_thing_to_plot="reduced_collector_temperature",
        plot_title="Electrical efficiency against reduced temperature",
        disable_lines=True,
        plot_trendline=True,
        first_axis_y_limits=[0.08, 0.14],
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

    # Plot the portion-covered affect on an unglazed Ilaria.
    reduced_data: Dict[str, Any] = collections.defaultdict(dict)
    thermal_efficiency_labels: Set[str] = set()
    electrical_efficiency_labels: Set[str] = set()
    for key, sub_dict in data.items():
        if "ilaria_unglazed_pc_" not in key:
            continue
        portion_covered_match = re.match(PORTION_COVERED_REGEX, key)
        if portion_covered_match is None:
            sys.exit(1)
        portion_covered = float(
            f"{portion_covered_match.group('first_digit')}.{portion_covered_match.group('second_digit')}"
        )
        # Only plot significant portions covered.
        if portion_covered not in {1.0, 0.6, 0.3}:
            continue
        for sub_key, value in sub_dict.items():
            # Fetch the portion covered.
            # Store the specific thermal efficiency.
            reduced_data[sub_key][f"thermal efficiency p.c.={portion_covered}"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency p.c.={portion_covered}")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][
                f"electrical efficiency p.c.={portion_covered}"
            ] = value["electrical_efficiency"]
            electrical_efficiency_labels.add(
                f"electrical efficiency p.c.={portion_covered}"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="ilaria_unglazed_pc",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the portion-covered affect on unglazed autotherm.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    for key, sub_dict in data.items():
        if "autotherm_unglazed_pc_" not in key:
            continue
        portion_covered_match = re.match(PORTION_COVERED_REGEX, key)
        if portion_covered_match is None:
            sys.exit(1)
        portion_covered = float(
            f"{portion_covered_match.group('first_digit')}.{portion_covered_match.group('second_digit')}"
        )
        # Only plot significant portions covered.
        if portion_covered not in {1.0, 0.6, 0.3}:
            continue
        for sub_key, value in sub_dict.items():
            # Fetch the portion covered.
            # Store the specific thermal efficiency.
            reduced_data[sub_key][f"thermal efficiency p.c.={portion_covered}"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency p.c.={portion_covered}")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][
                f"electrical efficiency p.c.={portion_covered}"
            ] = value["electrical_efficiency"]
            electrical_efficiency_labels.add(
                f"electrical efficiency p.c.={portion_covered}"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="autotherm_unglazed_pc",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the portion-covered affect on single-glazed Ilaria.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    for key, sub_dict in data.items():
        if "ilaria_single_glazed_pc_" not in key:
            continue
        portion_covered_match = re.match(PORTION_COVERED_REGEX, key)
        if portion_covered_match is None:
            sys.exit(1)
        portion_covered = float(
            f"{portion_covered_match.group('first_digit')}.{portion_covered_match.group('second_digit')}"
        )
        # Only plot significant portions covered.
        if portion_covered not in {1.0, 0.6, 0.3}:
            continue
        for sub_key, value in sub_dict.items():
            # Fetch the portion covered.
            # Store the specific thermal efficiency.
            reduced_data[sub_key][f"thermal efficiency p.c.={portion_covered}"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency p.c.={portion_covered}")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][
                f"electrical efficiency p.c.={portion_covered}"
            ] = value["electrical_efficiency"]
            electrical_efficiency_labels.add(
                f"electrical efficiency p.c.={portion_covered}"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="ilaria_single_glazed_pc",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the portion-covered affect on single-glazed autotherm.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    for key, sub_dict in data.items():
        if "autotherm_single_glazed_pc_" not in key:
            continue
        portion_covered_match = re.match(PORTION_COVERED_REGEX, key)
        if portion_covered_match is None:
            sys.exit(1)
        portion_covered = float(
            f"{portion_covered_match.group('first_digit')}.{portion_covered_match.group('second_digit')}"
        )
        # Only plot significant portions covered.
        if portion_covered not in {1.0, 0.6, 0.3}:
            continue
        for sub_key, value in sub_dict.items():
            # Fetch the portion covered.
            # Store the specific thermal efficiency.
            reduced_data[sub_key][f"thermal efficiency p.c.={portion_covered}"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency p.c.={portion_covered}")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][
                f"electrical efficiency p.c.={portion_covered}"
            ] = value["electrical_efficiency"]
            electrical_efficiency_labels.add(
                f"electrical efficiency p.c.={portion_covered}"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="autotherm_single_glazed_pc",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the portion-covered affect on double-glazed Ilaria.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    for key, sub_dict in data.items():
        if "ilaria_double_glazed_pc_" not in key:
            continue
        portion_covered_match = re.match(PORTION_COVERED_REGEX, key)
        if portion_covered_match is None:
            sys.exit(1)
        portion_covered = float(
            f"{portion_covered_match.group('first_digit')}.{portion_covered_match.group('second_digit')}"
        )
        # Only plot significant portions covered.
        if portion_covered not in {1.0, 0.6, 0.3}:
            continue
        for sub_key, value in sub_dict.items():
            # Fetch the portion covered.
            # Store the specific thermal efficiency.
            reduced_data[sub_key][f"thermal efficiency p.c.={portion_covered}"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency p.c.={portion_covered}")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][
                f"electrical efficiency p.c.={portion_covered}"
            ] = value["electrical_efficiency"]
            electrical_efficiency_labels.add(
                f"electrical efficiency p.c.={portion_covered}"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="ilaria_double_glazed_pc",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the portion-covered affect on double-glazed autotherm.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    for key, sub_dict in data.items():
        if "autotherm_double_glazed_pc_" not in key:
            continue
        portion_covered_match = re.match(PORTION_COVERED_REGEX, key)
        if portion_covered_match is None:
            sys.exit(1)
        portion_covered = float(
            f"{portion_covered_match.group('first_digit')}.{portion_covered_match.group('second_digit')}"
        )
        # Only plot significant portions covered.
        if portion_covered not in {1.0, 0.6, 0.3}:
            continue
        for sub_key, value in sub_dict.items():
            # Fetch the portion covered.
            # Store the specific thermal efficiency.
            reduced_data[sub_key][f"thermal efficiency p.c.={portion_covered}"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency p.c.={portion_covered}")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][
                f"electrical efficiency p.c.={portion_covered}"
            ] = value["electrical_efficiency"]
            electrical_efficiency_labels.add(
                f"electrical efficiency p.c.={portion_covered}"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="autotherm_double_glazed_pc",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the glazing effect on a fully covered Ilaria panel.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    ilaria_fully_covered_regex = re.compile(r"ilaria_(?P<glazing>[^0-9]*)_pc_1_0_.*")
    for key, sub_dict in data.items():
        ilaria_glazing_match = re.match(ilaria_fully_covered_regex, key)
        if ilaria_glazing_match is None:
            continue
        glazing = ilaria_glazing_match.group("glazing")
        for sub_key, value in sub_dict.items():
            reduced_data[sub_key][f"{glazing} thermal efficiency"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"{glazing} thermal efficiency")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][f"{glazing} electrical efficiency"] = value[
                "electrical_efficiency"
            ]
            electrical_efficiency_labels.add(f"{glazing} electrical efficiency")
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="ilaria_pc_1_glazing_comparison",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the glazing effect on a fully covered autotherm.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    autotherm_fully_covered_regex = re.compile(
        r"autotherm_(?P<glazing>[^0-9]*)_pc_1_0_.*"
    )
    for key, sub_dict in data.items():
        autotherm_glazing_match = re.match(autotherm_fully_covered_regex, key)
        if autotherm_glazing_match is None:
            continue
        glazing = autotherm_glazing_match.group("glazing")
        for sub_key, value in sub_dict.items():
            reduced_data[sub_key][f"{glazing} thermal efficiency"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"{glazing} thermal efficiency")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][f"{glazing} electrical efficiency"] = value[
                "electrical_efficiency"
            ]
            electrical_efficiency_labels.add(f"{glazing} electrical efficiency")
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="autotherm_pc_1_glazing_comparison",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the mass-flow-rate effect on a single-glazed Ilaria.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    ilaria_mass_flow_rate_regex = re.compile(
        r"ilaria.*single_glazed.*_(?P<first_digit>[0-9]*)_(?P<second_digit>[0-9])_litres_per_hour_.*"
    )
    for key, sub_dict in data.items():
        ilaria_mass_flow_rate_match = re.match(ilaria_mass_flow_rate_regex, key)
        if ilaria_mass_flow_rate_match is None:
            continue
        mass_flow_rate = round(
            int(ilaria_mass_flow_rate_match.group("first_digit"))
            + 0.1 * (int(ilaria_mass_flow_rate_match.group("second_digit"))),
            3,
        )
        if int(mass_flow_rate) == mass_flow_rate:
            mass_flow_rate = int(mass_flow_rate)
        # Only include important mass-flow rates.
        if mass_flow_rate not in [
            round(100, 3),
            round(90, 3),
            round(80, 3),
            round(70, 3),
            round(60, 3),
            round(50, 3),
            round(40, 3),
            round(30, 3),
            round(20, 3),
            round(10, 3),
            round(8, 3),
            round(6, 3),
            round(2, 3),
            round(1, 3),
            round(0.8, 3),
            round(0.6, 3),
            round(0.4, 3),
            round(0.2, 3),
            round(0.1, 3),
        ]:
            continue
        for sub_key, value in sub_dict.items():
            reduced_data[sub_key][f"thermal efficiency {mass_flow_rate}L/h"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency {mass_flow_rate}L/h")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][f"electrical efficiency {mass_flow_rate}L/h"] = value[
                "electrical_efficiency"
            ]
            electrical_efficiency_labels.add(
                f"electrical efficiency {mass_flow_rate}L/h"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    reduced_data.pop("20.0")

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="ilaria_single_glazed_mass_flow_rate_comparison",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the mass-flow-rate effect on a single-glazed autotherm.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    autotherm_mass_flow_rate_regex = re.compile(
        r"autotherm.*single_glazed.*_(?P<first_digit>[0-9]*)_(?P<second_digit>[0-9])_litres_per_hour_.*"
    )
    for key, sub_dict in data.items():
        autotherm_mass_flow_rate_match = re.match(autotherm_mass_flow_rate_regex, key)
        if autotherm_mass_flow_rate_match is None:
            continue
        mass_flow_rate = round(
            int(autotherm_mass_flow_rate_match.group("first_digit"))
            + 0.1 * (int(autotherm_mass_flow_rate_match.group("second_digit"))),
            3,
        )
        if int(mass_flow_rate) == mass_flow_rate:
            mass_flow_rate = int(mass_flow_rate)
        # Only include important mass-flow rates.
        if mass_flow_rate not in [
            round(100, 3),
            round(90, 3),
            round(80, 3),
            round(70, 3),
            round(60, 3),
            round(50, 3),
            round(40, 3),
            round(30, 3),
            round(20, 3),
            round(10, 3),
            round(8, 3),
            round(6, 3),
            round(2, 3),
            round(1, 3),
            round(0.8, 3),
            round(0.6, 3),
            round(0.4, 3),
            round(0.2, 3),
            round(0.1, 3),
        ]:
            continue
        for sub_key, value in sub_dict.items():
            reduced_data[sub_key][f"thermal efficiency {mass_flow_rate}L/h"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency {mass_flow_rate}L/h")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][f"electrical efficiency {mass_flow_rate}L/h"] = value[
                "electrical_efficiency"
            ]
            electrical_efficiency_labels.add(
                f"electrical efficiency {mass_flow_rate}L/h"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="autotherm_single_glazed_mass_flow_rate_comparison",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the mass-flow-rate effect on a double-glazed Ilaria.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    ilaria_mass_flow_rate_regex = re.compile(
        r"ilaria.*double_glazed.*_(?P<first_digit>[0-9]*)_(?P<second_digit>[0-9])_litres_per_hour_.*"
    )
    for key, sub_dict in data.items():
        ilaria_mass_flow_rate_match = re.match(ilaria_mass_flow_rate_regex, key)
        if ilaria_mass_flow_rate_match is None:
            continue
        mass_flow_rate = round(
            int(ilaria_mass_flow_rate_match.group("first_digit"))
            + 0.1 * (int(ilaria_mass_flow_rate_match.group("second_digit"))),
            3,
        )
        if int(mass_flow_rate) == mass_flow_rate:
            mass_flow_rate = int(mass_flow_rate)
        # Only include important mass-flow rates.
        if mass_flow_rate not in [
            round(100, 3),
            round(90, 3),
            round(80, 3),
            round(70, 3),
            round(60, 3),
            round(50, 3),
            round(40, 3),
            round(30, 3),
            round(20, 3),
            round(10, 3),
            round(8, 3),
            round(6, 3),
            round(2, 3),
            round(1, 3),
            round(0.8, 3),
            round(0.6, 3),
            round(0.4, 3),
            round(0.2, 3),
            round(0.1, 3),
        ]:
            continue
        for sub_key, value in sub_dict.items():
            reduced_data[sub_key][f"thermal efficiency {mass_flow_rate}L/h"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency {mass_flow_rate}L/h")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][f"electrical efficiency {mass_flow_rate}L/h"] = value[
                "electrical_efficiency"
            ]
            electrical_efficiency_labels.add(
                f"electrical efficiency {mass_flow_rate}L/h"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="ilaria_double_glazed_mass_flow_rate_comparison",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the mass-flow-rate effect on a single-glazed autotherm.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    autotherm_mass_flow_rate_regex = re.compile(
        r"autotherm.*double_glazed.*_(?P<first_digit>[0-9]*)_(?P<second_digit>[0-9])_litres_per_hour_.*"
    )
    for key, sub_dict in data.items():
        autotherm_mass_flow_rate_match = re.match(autotherm_mass_flow_rate_regex, key)
        if autotherm_mass_flow_rate_match is None:
            continue
        mass_flow_rate = round(
            int(autotherm_mass_flow_rate_match.group("first_digit"))
            + 0.1 * (int(autotherm_mass_flow_rate_match.group("second_digit"))),
            3,
        )
        if int(mass_flow_rate) == mass_flow_rate:
            mass_flow_rate = int(mass_flow_rate)
        # Only include important mass-flow rates.
        if mass_flow_rate not in [
            round(100, 3),
            round(90, 3),
            round(80, 3),
            round(70, 3),
            round(60, 3),
            round(50, 3),
            round(40, 3),
            round(30, 3),
            round(20, 3),
            round(10, 3),
            round(8, 3),
            round(6, 3),
            round(2, 3),
            round(1, 3),
            round(0.8, 3),
            round(0.6, 3),
            round(0.4, 3),
            round(0.2, 3),
            round(0.1, 3),
        ]:
            continue
        for sub_key, value in sub_dict.items():
            reduced_data[sub_key][f"thermal efficiency {mass_flow_rate}L/h"] = value[
                "thermal_efficiency"
            ]
            thermal_efficiency_labels.add(f"thermal efficiency {mass_flow_rate}L/h")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][f"electrical efficiency {mass_flow_rate}L/h"] = value[
                "electrical_efficiency"
            ]
            electrical_efficiency_labels.add(
                f"electrical efficiency {mass_flow_rate}L/h"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="autotherm_double_glazed_mass_flow_rate_comparison",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )

    plt.close("all")

    # Plot the number-of-pipes effect on a single-glazed Ilaria.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    ilaria_pipes_regex = re.compile(
        r"ilaria_(?P<pipes>[0-9]*)_pipes_(?P<glazing>.*)_pc.*"
    )
    for key, sub_dict in data.items():
        ilaria_pipes_match = re.match(ilaria_pipes_regex, key)
        if ilaria_pipes_match is None:
            continue
        pipes = int(ilaria_pipes_match.group("pipes"))
        glazing = ilaria_pipes_match.group("glazing")
        # if glazing == "unglazed":
        #     continue
        # Only include important mass-flow rates.
        for sub_key, value in sub_dict.items():
            reduced_data[sub_key][
                f"{glazing} thermal efficiency {pipes} pipes"
            ] = value["thermal_efficiency"]
            thermal_efficiency_labels.add(f"{glazing} thermal efficiency {pipes} pipes")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][
                f"{glazing} electrical efficiency {pipes} pipes"
            ] = value["electrical_efficiency"]
            electrical_efficiency_labels.add(
                f"{glazing} electrical efficiency {pipes} pipes"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="ilaria_pc_1_pipes_glazing_comparison",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")

    # Plot the number-of-pipes effect on a single-glazed Autotherm.
    reduced_data = collections.defaultdict(dict)
    thermal_efficiency_labels = set()
    electrical_efficiency_labels = set()
    autotherm_pipes_regex = re.compile(
        r"autotherm_(?P<pipes>[0-9]*)_pipes_(?P<glazing>.*)_pc.*"
    )
    for key, sub_dict in data.items():
        autotherm_pipes_match = re.match(autotherm_pipes_regex, key)
        if autotherm_pipes_match is None:
            continue
        pipes = int(autotherm_pipes_match.group("pipes"))
        glazing = autotherm_pipes_match.group("glazing")
        if glazing == "unglazed":
            continue
        # Only include important mass-flow rates.
        for sub_key, value in sub_dict.items():
            reduced_data[sub_key][
                f"{glazing} thermal efficiency {pipes} pipes"
            ] = value["thermal_efficiency"]
            thermal_efficiency_labels.add(f"{glazing} thermal efficiency {pipes} pipes")
            # Store the specific electrical efficiency.
            reduced_data[sub_key][
                f"{glazing} electrical efficiency {pipes} pipes"
            ] = value["electrical_efficiency"]
            electrical_efficiency_labels.add(
                f"{glazing} electrical efficiency {pipes} pipes"
            )
            # Store a copy of the reduced temperature
            reduced_data[sub_key]["reduced_collector_temperature"] = value[
                "reduced_collector_temperature"
            ]

    _efficiency_plots(
        data=reduced_data,
        electrical_efficiency_labels=sorted(electrical_efficiency_labels),
        logger=logger,
        prefix="autotherm_pc_1_pipes_glazing_comparison",
        thermal_efficiency_labels=sorted(thermal_efficiency_labels),
    )
    plt.close("all")


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
