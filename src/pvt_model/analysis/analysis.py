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
from typing import Any, List, Dict, Optional, Union

import yaml

from matplotlib import pyplot as plt

try:
    from ..__utils__ import get_logger
    from ..pvt_system.constants import (  # pylint: disable=unused-import
        HEAT_CAPACITY_OF_WATER,
    )
    from ..pvt_system.physics_utils import reduced_temperature
    from .__utils__ import (
        GraphDetail,
        load_model_data,
        plot_figure,
        plot_two_dimensional_figure,
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


def analyse_coupled_dynamic_data(
    data: Dict[Any, Any], logger: Logger, skip_2d_plots: bool
) -> None:
    """
    Carry out analysis on a set of dynamic data.

    :param data:
        The data to analyse.

    :param logger:
        The logger to use for the analysis run.

    :param skip_2d_plots:
        Whether to skip the 2D plots (True) or include them (False).

    """

    logger.info("Beginning analysis of coupled dynamic data set.")

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
        "ambient_temperature",
        data,
        first_axis_things_to_plot=["ambient_temperature", "sky_temperature"],
        first_axis_label="Temperature / deg C",
        first_axis_y_limits=[0, 65],
    )

    # Plot Figure 6a: Panel-related Temperatures
    plot_figure(
        "panel_temperature",
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
        "tank_temperature",
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
        "stream_temperature",
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
    if not skip_2d_plots:
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
        "pvt_collector_temperature",
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

    for temperature in data.keys():
        temperature_string = str(round(float(temperature), 2)).replace(".", "_")

        if not skip_2d_plots:
            # Glass Temperatures
            try:
                logger.info(
                    "Plotting 3D upper glass profile at %s degC.", temperature_string
                )
                plot_two_dimensional_figure(
                    "steady_state_upper_glass_layer_{}degC_input".format(
                        temperature_string
                    ),
                    logger,
                    data,
                    axis_label="Temperature / deg C",
                    entry_number=temperature,
                    plot_title="Upper glass layer temperature with {} K input HTF".format(
                        round(float(temperature), 2)
                    ),
                    thing_to_plot="layer_temperature_map_upper_glass",
                )
            except TypeError:
                logger.info(
                    "Upper-glass temperature profile could not be plotted due to no data."
                )

            # Glass Temperatures
            try:
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
            except TypeError:
                print("Glass temperature profile could not be plotted due to no data.")

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

    # Parse the thermal-efficiency data.
    with open(
        os.path.join("system_data", "steady_state_data", STEADY_STATE_DATA_FILE_NAME),
        "r",
    ) as f:  #
        experimental_steady_state_data = yaml.safe_load(f)

    # Post-process this data.
    for entry in experimental_steady_state_data:
        entry["reduced_temperature"] = reduced_temperature(
            entry["ambient_temperature"],
            entry["average_bulk_water_temperature"],
            entry["irradiance"],
        )

    # Thermal efficiency plot.
    logger.info("Plotting thermal efficiency against the reduced temperature.")

    # Plot the experimental data.
    _, ax1 = plt.subplots()
    ax1.scatter(
        [entry["reduced_temperature"] for entry in experimental_steady_state_data],
        [entry["thermal_efficiency"] for entry in experimental_steady_state_data],
        marker="s",
    )

    # Add the model data.
    plot_figure(
        "thermal_efficiency_against_reduced_temperature",
        data,
        first_axis_things_to_plot=["thermal_efficiency"],
        first_axis_label="Thermal efficiency",
        x_axis_label="Reduced temperature / K m^2 / W",
        x_axis_thing_to_plot="reduced_collector_temperature",
        plot_title="Thermal efficiency against reduced temperature",
        disable_lines=True,
        override_axis=ax1,
    )

    # Collector temperature gain plot.
    logger.info(
        "Plotting collector temperature gain against the input HTF temperature."
    )

    # Plot the experimental data.
    _, ax1 = plt.subplots()
    ax1.scatter(
        [
            entry["collector_input_temperature"]
            for entry in experimental_steady_state_data
        ],
        [
            entry["collector_temperature_gain"]
            for entry in experimental_steady_state_data
        ],
        marker="s",
    )

    # Add the model data.
    plot_figure(
        "collector_tempreature_gain_against_input_temperature",
        data,
        first_axis_things_to_plot=["collector_temperature_gain"],
        first_axis_label="Collector temperature gain / K",
        x_axis_label="Collector input temperature / degC",
        use_data_keys_as_x_axis=True,
        plot_title="Collector temperature gain against input temperature",
        disable_lines=True,
        override_axis=ax1,
    )

    # Plot the electrical efficiency against the reduced temperature.
    plot_figure(
        "electrical_efficiency_against_reduced_temperature",
        data,
        first_axis_things_to_plot=["electrical_efficiency"],
        first_axis_label="Electrical efficiency",
        x_axis_label="Reduced temperature / K m^2 / W",
        x_axis_thing_to_plot="reduced_collector_temperature",
        plot_title="Electrical efficiency against reduced temperature",
        disable_lines=True,
    )


def analyse_decoupled_dynamic_data(data: Dict[Any, Any], logger: Logger) -> None:
    """
    Carry out analysis on a decoupled dyanmic set of data.

    :param data:
        The data to analyse.

    :param logger:
        The logger to use for the run.

    """

    logger.info("Beginning analysis of a decoupled dynamic data set.")

    # * Reduce the resolution of the data.
    data = _reduce_data(data, GRAPH_DETAIL, logger)
    logger.info(
        "Data successfully reduced to %s graph detail level.", GRAPH_DETAIL.name
    )

    # * Create new data values where needed.
    data = _post_process_data(data)
    logger.info("Post-processing of data complete.")

    # Clip out the data points up to 10 minutes in.
    data = {key: value for key, value in data.items() if int(key) >= 30}

    # Plot output temperature and irradiance.
    plot_figure(
        "collector_output_response",
        data,
        first_axis_things_to_plot=[
            "collector_output_temperature",
            "collector_input_temperature",
            "ambient_temperature",
        ],
        first_axis_label="Collector Output Temperature / deg C",
        first_axis_y_limits=[15, 25],
        second_axis_things_to_plot=[
            "solar_irradiance",
        ],
        second_axis_label="Solar Irradiance / W/m^2",
        second_axis_y_limits=[0, 1000],
    )


def analyse(
    data_file_name: str,
    show_output: Optional[bool] = False,
    skip_2d_plots: Optional[bool] = False,
) -> None:
    """
    The main method for the analysis module.

    :param data_file_name:
        The path to the data file to analyse.

    :param show_output:
        Whether to show the output files generated.

    :param skip_2d_plots:
        Whether to skip the 2D plots (True) of include them (False).

    """

    # * Set up the logger
    logger = get_logger(True, "pvt_analysis", True)

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
    if data_type in (DYNAMIC_DATA_TYPE, f"{COUPLED_DATA_TYPE}_{DYNAMIC_DATA_TYPE}"):
        analyse_coupled_dynamic_data(data, logger, skip_2d_plots)
    elif data_type in (
        STEADY_STATE_DATA_TYPE,
        f"{DECOUPLED_DATA_TYPE}_{STEADY_STATE_DATA_TYPE}",
    ):
        analyse_decoupled_steady_state_data(data, logger, skip_2d_plots)
    elif data_type == f"{DECOUPLED_DATA_TYPE}_{DYNAMIC_DATA_TYPE}":
        analyse_decoupled_dynamic_data(data, logger)
    else:
        logger.error("Data type was neither 'dynamic' nor 'steady_state'. Exiting...")
        sys.exit(1)

    logger.info("Analysis complete - all figures saved successfully.")

    if show_output:
        plt.show()


if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])
    analyse(
        parsed_args.data_file_name, parsed_args.show_output, parsed_args.skip_2d_plots
    )
