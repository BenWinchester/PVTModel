#!/usr/bin/python3.7
########################################################################################
# __main__.py - The main module for the higher-level PV-T model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The main module for the PV-T model.

The model is run from here for several runs as determined by command-line arguments.

"""

import dataclasses
import os
import sys

from argparse import Namespace
from functools import partial
from logging import Logger
from multiprocessing import Pool
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import json
import yaml

from . import argparser

from .__utils__ import (
    BColours,
    CarbonEmissions,
    COARSE_RUN_RESOLUTION,
    FileType,
    fourier_number,
    get_logger,
    INITIAL_CONDITION_PRECISION,
    LOGGER_NAME,
    MissingParametersError,
    OperatingMode,
    ProgrammerJudgementFault,
    read_yaml,
    SystemData,
    TemperatureName,
    TotalPowerData,
)

from .analysis import analysis

from .pvt_system_model import index_handler, tank
from .pvt_system_model.pvt_panel import pvt
from .pvt_system_model.__main__ import main as pvt_system_model_main
from .pvt_system_model.__utils__ import DivergentSolutionError

from .pvt_system_model.constants import (
    DEFAULT_SYSTEM_TEMPERATURE,
    DENSITY_OF_WATER,
    HEAT_CAPACITY_OF_WATER,
    THERMAL_CONDUCTIVITY_OF_WATER,
    ZERO_CELCIUS_OFFSET,
)
from .pvt_system_model.process_pvt_system_data import (
    hot_water_tank_from_path,
    pvt_panel_from_path,
)


def _get_system_fourier_numbers(
    hot_water_tank: Optional[tank.Tank], pvt_panel: pvt.PVT, resolution: float
) -> Dict[TemperatureName, float]:
    """
    Determine the Fourier numbers of the various system components.

    :param hot_water_tank:
        A :class:`tank.Tank` instance representing the hot-water tank in the system.

    :param pvt_panel:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The resolution being used for the model.

    :return:
        The Fourier coefficients for the various components being modelled.

    """

    # Determine the Fourier coefficients of the panel's layers.
    fourier_number_map: Dict[TemperatureName, float] = dict()
    if pvt_panel.glass is not None:
        fourier_number_map[TemperatureName.glass] = round(
            fourier_number(
                pvt_panel.glass.thickness,
                pvt_panel.glass.conductivity,
                pvt_panel.glass.density,
                pvt_panel.glass.heat_capacity,
                resolution,
            ),
            2,
        )
    if pvt_panel.pv is not None:
        fourier_number_map[TemperatureName.pv] = round(
            fourier_number(
                pvt_panel.pv.thickness,
                pvt_panel.pv.conductivity,
                pvt_panel.pv.density,
                pvt_panel.pv.heat_capacity,
                resolution,
            ),
            2,
        )
    fourier_number_map[TemperatureName.absorber] = round(
        fourier_number(
            pvt_panel.absorber.thickness,
            pvt_panel.absorber.conductivity,
            pvt_panel.absorber.density,
            pvt_panel.absorber.heat_capacity,
            resolution,
        ),
        2,
    )
    fourier_number_map[TemperatureName.htf] = round(
        fourier_number(
            pvt_panel.absorber.inner_pipe_diameter,
            THERMAL_CONDUCTIVITY_OF_WATER,
            DENSITY_OF_WATER,
            pvt_panel.absorber.htf_heat_capacity,
            resolution,
        ),
        2,
    )
    if hot_water_tank is not None:
        fourier_number_map[TemperatureName.tank] = round(
            fourier_number(
                hot_water_tank.diameter,
                THERMAL_CONDUCTIVITY_OF_WATER,
                DENSITY_OF_WATER,
                HEAT_CAPACITY_OF_WATER,
                resolution,
            ),
            5,
        )

    return fourier_number_map


def _determine_fourier_numbers(
    hot_water_tank: Optional[tank.Tank],
    logger: Logger,
    parsed_args: Namespace,
    pvt_panel: pvt.PVT,
) -> None:
    """
    Determines, prints, and logs the various Fourier numbers.

    :param hot_water_tank:
        A :class:`tank.Tank` instance representing the hot-water tank being modelled.

    :param logger:
        The logger being used for the run.

    :param parsed_args:
        The parsed commnand-line arguments.

    :param pvt_panel:
        The pvt panel, represented as a :class:`pvt_panel.pvt.PVT` instance.

    """

    # Determine the Fourier numbers.
    fourier_number_map = _get_system_fourier_numbers(
        hot_water_tank, pvt_panel, parsed_args.resolution
    )

    logger.info(
        "Fourier numbers determined:\n%s\n%s",
        "|".join(
            [
                " {}{}".format(
                    key.name,
                    " "
                    * (
                        max(
                            [len(key.name) for key in fourier_number_map]
                            + [len(str(value)) for value in fourier_number_map.values()]
                        )
                        - len(key.name)
                        + 1
                    ),
                )
                for key in fourier_number_map
            ]
        ),
        "|".join(
            [
                " {}{}".format(
                    value,
                    " "
                    * (
                        max(
                            [len(key.name) for key in fourier_number_map]
                            + [len(str(value)) for value in fourier_number_map.values()]
                        )
                        - len(str(value))
                        + 1
                    ),
                )
                for value in fourier_number_map.values()
            ]
        ),
    )
    print(
        "Fourier numbers determined:\n{}\n{}".format(
            "|".join(
                [
                    " {}{}".format(
                        key.name,
                        " "
                        * (
                            max(
                                [len(key.name) for key in fourier_number_map]
                                + [
                                    len(str(value))
                                    for value in fourier_number_map.values()
                                ]
                            )
                            - len(key.name)
                            + 1
                        ),
                    )
                    for key in fourier_number_map
                ]
            ),
            "|".join(
                [
                    " {}{}".format(
                        value,
                        " "
                        * (
                            max(
                                [len(key.name) for key in fourier_number_map]
                                + [
                                    len(str(value))
                                    for value in fourier_number_map.values()
                                ]
                            )
                            - len(str(value))
                            + 1
                        ),
                    )
                    for value in fourier_number_map.values()
                ]
            ),
        )
    )


def _determine_consistent_conditions(
    number_of_pipes: int,
    layers: Set[TemperatureName],
    logger: Logger,
    operating_mode: OperatingMode,
    parsed_args: Namespace,
    pvt_panel: pvt.PVT,
    *,
    override_ambient_temperature: Optional[float] = None,
    override_collector_input_temperature: Optional[float] = None,
    override_irradiance: Optional[float] = None,
    override_wind_speed: Optional[float] = None,
    resolution: int = COARSE_RUN_RESOLUTION,
    run_depth: int = 1,
    running_system_temperature_vector: Optional[List[float]] = None,
) -> Tuple[Optional[List[float]], Dict[float, SystemData]]:
    """
    Determines the initial system temperatures for the run.

    :param number_of_pipes:
        The number of pipes on the base of the hot-water absorber.

    :param layers:
        The layer being used for the run.

    :param logger:
        The logger for the run.

    :param operating_mode:
        The operating mode for the run.

    :param parsed_args:
        The parsed command-line arguments.

    :param pvt_panel:
        The :class:`pvt.PVT` instance representing the pvt panel being modelled.

    :param override_ambient_tempearture:
        If specified, this can be used as a value to override the weather forecaster's
        inbuilt ambient temperature.

    :param resolution:
        The resolution for the run, measured in seconds.

    :param run_depth:
        The depth of the recursion.

    :param running_system_temperature_vector:
        The current vector for the system temperatures, used to commence a run and
        compare with the previous run.

    :return:
        A `tuple` containing:
        - the initial system temperature which fits within the desired resolution, if
          applicable, else `None`;
        - the system data from the sun which satisfies the desired consistency.

    """

    # Fetch the initial temperature conditions if not passed in:
    if running_system_temperature_vector is None:
        if operating_mode.coupled:
            running_system_temperature_vector = [
                DEFAULT_SYSTEM_TEMPERATURE
            ] * index_handler.num_temperatures(pvt_panel)
        else:
            running_system_temperature_vector = [DEFAULT_SYSTEM_TEMPERATURE] * (
                index_handler.num_temperatures(pvt_panel) - 3
            )

    # Call the model to generate the output of the run.
    logger.info("Running the model. Run number %s.", run_depth)
    final_temperature_vector, system_data = pvt_system_model_main(
        parsed_args.average_irradiance,
        parsed_args.cloud_efficacy_factor,
        parsed_args.disable_logging,
        parsed_args.exchanger_data_file,
        parsed_args.initial_month,
        running_system_temperature_vector,
        layers,
        parsed_args.location,
        operating_mode,
        parsed_args.portion_covered,
        parsed_args.pvt_data_file,
        resolution,
        not parsed_args.skip_2d_output,
        parsed_args.tank_data_file,
        parsed_args.use_pvgis,
        parsed_args.verbose,
        parsed_args.x_resolution,
        parsed_args.y_resolution,
        days=parsed_args.days,
        minutes=parsed_args.minutes,
        months=parsed_args.months,
        override_ambient_temperature=override_ambient_temperature,
        override_collector_input_temperature=override_collector_input_temperature,
        override_irradiance=override_irradiance,
        override_mass_flow_rate=parsed_args.mass_flow_rate,
        override_wind_speed=override_wind_speed,
        run_number=run_depth,
        start_time=parsed_args.start_time,
    )

    # If in verbose mode, output average, min, and max temperatures.
    if parsed_args.verbose:
        _output_temperature_info(logger, parsed_args, system_data)

    # If all the temperatures are within the desired limit, return the temperatures.
    if all(
        abs(final_temperature_vector - running_system_temperature_vector)
        <= INITIAL_CONDITION_PRECISION
    ):
        logger.info(
            "Initial temperatures consistent. Max difference: %sK",
            max(abs(final_temperature_vector - running_system_temperature_vector)),
        )
        return final_temperature_vector.tolist(), system_data

    if operating_mode.decoupled:
        logger.info("Steady-state run determined at convergence precision.")
        return final_temperature_vector.tolist(), system_data

    logger.info(
        "Initial temperatures not consistent. Max difference: %sK",
        max(abs(final_temperature_vector - running_system_temperature_vector)),
    )
    # Otherwise, call the method recursively.
    return _determine_consistent_conditions(
        number_of_pipes,
        layers,
        logger,
        operating_mode,
        parsed_args,
        pvt_panel,
        override_ambient_temperature=override_ambient_temperature,
        override_collector_input_temperature=override_collector_input_temperature,
        override_irradiance=override_irradiance,
        override_wind_speed=override_wind_speed,
        resolution=resolution,
        run_depth=run_depth + 1,
        running_system_temperature_vector=final_temperature_vector.tolist(),
    )


def _multiprocessing_determine_consistent_conditions(
    steady_state_run: Dict[Any, Any],
    *,
    number_of_pipes: int,
    layers: Set[TemperatureName],
    logger: Logger,
    operating_mode: OperatingMode,
    parsed_args: Namespace,
    pvt_panel: pvt.PVT,
) -> Dict[float, SystemData]:
    """
    Wrapper function around `determine_consistent_conditions` to enable multi-processing

    In order for multi-processing to occur, arguments need to be processed in such a way
    that the _determine_consistent_conditions function can be called using a single
    entry from a steady-state data file.

    :param steady_state_entry:
        A data entry from the steady-state data file.

    :param number_of_pipes:
        The number of pipes on the base of the hot-water absorber.

    :param layers:
        The layer being used for the run.

    :param logger:
        The logger for the run.

    :param operating_mode:
        The operating mode for the run.

    :param parsed_args:
        The parsed command-line arguments.

    :param pvt_panel:
        The :class:`pvt.PVT` instance representing the pvt panel being modelled.

    :return:
        The result of the function call to `_determine_consistent_conditions`.

    """

    _, system_data_entry = _determine_consistent_conditions(
        number_of_pipes,
        layers,
        logger,
        operating_mode,
        parsed_args,
        pvt_panel,
        override_ambient_temperature=steady_state_run["ambient_temperature"],
        override_collector_input_temperature=steady_state_run[
            "collector_input_temperature"
        ]
        + ZERO_CELCIUS_OFFSET,
        override_irradiance=steady_state_run["irradiance"],
        override_wind_speed=steady_state_run["wind_speed"],
    )

    return system_data_entry


def _print_temperature_info(
    average_temperature_map: Dict[str, float],
    logger: Logger,
    maximum_temperature_map: Dict[str, float],
    minimum_temperature_map: Dict[str, float],
    verbose_mode: bool = False,
) -> None:
    """
    Prints out the average temperatures to the console and logger.

    :param average_temperature_map:
        A mapping between temperature name, as a `str`, and the average temperature for
        the run.

    :param logger:
        The logger used.

    :param maximum_temperature_run:
        A mapping between temperature name, as a `str`, and the maximum temperature for
        the run.

    :param minimum_temperature_run:
        A mapping between temperature name, as a `str`, and the minimum temperature for
        the run.

    :param verbose_mode:
        If True, then colouring will be done of the output to indicate that it is
        verbose-mode specific.

    """

    logger.info(
        "Average temperatures for the run in degC:\n%s\n%s",
        "|".join(
            [
                " {}{}".format(
                    key,
                    " "
                    * (
                        max([len(key) for key in average_temperature_map])
                        - len(key)
                        + 1
                    ),
                )
                for key in average_temperature_map
            ]
        ),
        "|".join(
            [
                " {}{}".format(
                    value,
                    " "
                    * (
                        max([len(key) for key in average_temperature_map])
                        - len(str(value))
                        + 1
                    ),
                )
                for value in average_temperature_map.values()
            ]
        ),
    )
    print(
        "{}Average temperatures for the run in degC:\n{}\n{}{}".format(
            BColours.OKTEAL if verbose_mode else "",
            "|".join(
                [
                    " {}{}".format(
                        key,
                        " "
                        * (
                            max([len(key) for key in average_temperature_map])
                            - len(key)
                            + 1
                        ),
                    )
                    for key in average_temperature_map
                ]
            ),
            "|".join(
                [
                    " {}{}".format(
                        value,
                        " "
                        * (
                            max([len(key) for key in average_temperature_map])
                            - len(str(value))
                            + 1
                        ),
                    )
                    for value in average_temperature_map.values()
                ]
            ),
            BColours.ENDC,
        )
    )

    logger.info(
        "Maximum temperatures for the run in degC:\n%s\n%s",
        "|".join(
            [
                " {}{}".format(
                    key,
                    " "
                    * (
                        max([len(key) for key in maximum_temperature_map])
                        - len(key)
                        + 1
                    ),
                )
                for key in maximum_temperature_map
            ]
        ),
        "|".join(
            [
                " {}{}".format(
                    value,
                    " "
                    * (
                        max([len(key) for key in maximum_temperature_map])
                        - len(str(value))
                        + 1
                    ),
                )
                for value in maximum_temperature_map.values()
            ]
        ),
    )
    print(
        "{}Maximum temperatures for the run in degC:\n{}\n{}{}".format(
            BColours.OKTEAL if verbose_mode else "",
            "|".join(
                [
                    " {}{}".format(
                        key,
                        " "
                        * (
                            max([len(key) for key in maximum_temperature_map])
                            - len(key)
                            + 1
                        ),
                    )
                    for key in maximum_temperature_map
                ]
            ),
            "|".join(
                [
                    " {}{}".format(
                        value,
                        " "
                        * (
                            max([len(key) for key in maximum_temperature_map])
                            - len(str(value))
                            + 1
                        ),
                    )
                    for value in maximum_temperature_map.values()
                ]
            ),
            BColours.ENDC,
        )
    )

    logger.info(
        "Minimum temperatures for the run in degC:\n%s\n%s",
        "|".join(
            [
                " {}{}".format(
                    key,
                    " "
                    * (
                        max([len(key) for key in minimum_temperature_map])
                        - len(key)
                        + 1
                    ),
                )
                for key in minimum_temperature_map
            ]
        ),
        "|".join(
            [
                " {}{}".format(
                    value,
                    " "
                    * (
                        max([len(key) for key in minimum_temperature_map])
                        - len(str(value))
                        + 1
                    ),
                )
                for value in minimum_temperature_map.values()
            ]
        ),
    )
    print(
        "{}Minimum temperatures for the run in degC:\n{}\n{}{}".format(
            BColours.OKTEAL if verbose_mode else "",
            "|".join(
                [
                    " {}{}".format(
                        key,
                        " "
                        * (
                            max([len(key) for key in minimum_temperature_map])
                            - len(key)
                            + 1
                        ),
                    )
                    for key in minimum_temperature_map
                ]
            ),
            "|".join(
                [
                    " {}{}".format(
                        value,
                        " "
                        * (
                            max([len(key) for key in minimum_temperature_map])
                            - len(str(value))
                            + 1
                        ),
                    )
                    for value in minimum_temperature_map.values()
                ]
            ),
            BColours.ENDC,
        )
    )


def _output_temperature_info(
    logger: Logger, parsed_args: Namespace, system_data: Dict[float, SystemData]
) -> None:
    """
    Determines and prints information about the system temperatures.

    The average, minimum, and maximum temperatures of the various components are
    outputted to the console and the logs.

    :param logger:
        The logger used for the run.

    :param parsed_args:
        The parsed command-line arguments.

    :param system_data:
        The system data ouputted by the run.

    """

    # Determine the average, minimum, and maximum temperatures.
    average_temperature_map = {
        "pv": round(mean({entry.pv_temperature for entry in system_data.values()}), 3),
        "absorber": round(
            mean({entry.absorber_temperature for entry in system_data.values()}), 3
        ),
        "htf": round(
            mean({entry.bulk_water_temperature for entry in system_data.values()}),
            3,
        ),
    }
    maximum_temperature_map = {
        "pv": max({round(entry.pv_temperature, 3) for entry in system_data.values()}),
        "absorber": max(
            {round(entry.absorber_temperature, 3) for entry in system_data.values()}
        ),
        "htf": max(
            {round(entry.bulk_water_temperature, 3) for entry in system_data.values()}
        ),
    }
    minimum_temperature_map = {
        "pv": min({round(entry.pv_temperature, 3) for entry in system_data.values()}),
        "absorber": min(
            {round(entry.absorber_temperature, 3) for entry in system_data.values()}
        ),
        "htf": min(
            {round(entry.bulk_water_temperature, 3) for entry in system_data.values()}
        ),
    }

    if "dg" in parsed_args.layers:
        average_temperature_map["upper_glass"] = round(
            mean({entry.upper_glass_temperature for entry in system_data.values()}), 3  # type: ignore
        )
        maximum_temperature_map["upper_glass"] = max(
            {round(entry.upper_glass_temperature, 3) for entry in system_data.values()}  # type: ignore
        )
        minimum_temperature_map["upper_glass"] = min(
            {round(entry.upper_glass_temperature, 3) for entry in system_data.values()}  # type: ignore
        )

    if "g" in parsed_args.layers:
        average_temperature_map["glass"] = round(
            mean({entry.glass_temperature for entry in system_data.values()}), 3  # type: ignore
        )
        maximum_temperature_map["glass"] = max(
            {round(entry.glass_temperature, 3) for entry in system_data.values()}  # type: ignore
        )
        minimum_temperature_map["glass"] = min(
            {round(entry.glass_temperature, 3) for entry in system_data.values()}  # type: ignore
        )

    if not parsed_args.decoupled:
        average_temperature_map["tank"] = round(
            mean({entry.tank_temperature for entry in system_data.values()}), 3  # type: ignore
        )
        maximum_temperature_map["tank"] = max(
            {round(entry.tank_temperature, 3) for entry in system_data.values()}  # type: ignore
        )
        minimum_temperature_map["tank"] = min(
            {round(entry.tank_temperature, 3) for entry in system_data.values()}  # type: ignore
        )

    # Print these out to the console.
    logger.info("Average, minimum, and maximum temperatures determined.")
    _print_temperature_info(
        average_temperature_map,
        logger,
        maximum_temperature_map,
        minimum_temperature_map,
        True,
    )


def _save_data(
    file_type: FileType,
    logger: Logger,
    operating_mode: OperatingMode,
    output_file_name: str,
    system_data: Dict[float, SystemData],
    carbon_emissions: Optional[CarbonEmissions] = None,
    total_power_data: Optional[TotalPowerData] = None,
) -> None:
    """
    Save data when called. The data entry should be appended to the file.

    :param file_type:
        The file type that's being saved.

    :param logger:
        The logger used for the run.

    :param operating_mode:
        The operating mode for the run.

    :param output_file_name:
        The destination file name.

    :param system_data:
        The data to save.

    :param carbon_emissions:
        The carbon emissions data for the run.

    :param total_power_data:
        The total power data for the run.

    """

    # Convert the system data entry to JSON-readable format
    system_data_dict: Dict[Union[float, str], Union[str, Dict[str, Any]]] = {
        key: dataclasses.asdict(value) for key, value in system_data.items()
    }
    logger.info(
        "Saving data: System data successfully converted to json-readable format."
    )

    # If we're saving YAML data part-way through, then append to the file.
    if file_type == FileType.YAML:
        logger.info("Saving as YAML format.")
        with open(f"{output_file_name}.yaml", "a") as output_yaml_file:
            yaml.dump(
                system_data_dict,
                output_yaml_file,
            )

    # If we're dumping JSON, open the file, and append to it.
    if file_type == FileType.JSON:
        logger.info("Saving as JSON format.")
        # Append the total power and emissions data for the run.
        if total_power_data is not None:
            system_data_dict.update(dataclasses.asdict(total_power_data))  # type: ignore
            logger.info("Total power data updated.")
        if carbon_emissions is not None:
            system_data_dict.update(dataclasses.asdict(carbon_emissions))  # type: ignore
            logger.info("Carbon emissions updated.")

        # Append the data type for the run.
        system_data_dict["data_type"] = "{coupling}_{timing}".format(
            coupling="coupled" if operating_mode.coupled else "decoupled",
            timing="steady_state" if operating_mode.steady_state else "dynamic",
        )
        logger.info("Data type added successfully.")

        # Save the data
        # If this is the initial dump, then create the file.
        if not os.path.isfile(f"{output_file_name}.json"):
            logger.info("Attempting first dump to a non-existent file.")
            logger.info("Output file name: %s", output_file_name)
            with open(f"{output_file_name}.json", "w") as output_json_file:
                json.dump(
                    system_data_dict,
                    output_json_file,
                    indent=4,
                )
        else:
            with open(f"{output_file_name}.json", "r+") as output_json_file:
                # Read the data and append the current update.
                filedata = json.load(output_json_file)
                filedata.update(system_data_dict)
                # Overwrite the file with the updated data.
                output_json_file.seek(0)
                json.dump(
                    filedata,
                    output_json_file,
                    indent=4,
                )


def main(args) -> None:  # pylint: disable=too-many-branches
    """
    The main module for the code.

    :param args:
        The command-line arguments passed into the component.

    """

    # Parse the arguments passed in.
    parsed_args = argparser.parse_args(args)

    # Initialise logging.
    logger = get_logger(parsed_args.disable_logging, LOGGER_NAME, parsed_args.verbose)
    logger.info(
        "%s PVT model instantiated. %s\nCommand: %s", "=" * 20, "=" * 20, " ".join(args)
    )
    print(
        "PVT model instantiated{}.".format(
            f"{BColours.OKTEAL} in verbose mode{BColours.ENDC}"
            if parsed_args.verbose
            else ""
        )
    )

    # Check that all CLI args are valid.
    layers = argparser.check_args(
        parsed_args,
        logger,
        read_yaml(parsed_args.pvt_data_file)["absorber"]["number_of_pipes"],
    )

    # Parse the PVT system information and generate a PVT panel based on the args.
    pvt_panel = pvt_panel_from_path(
        layers,
        logger,
        parsed_args.mass_flow_rate,
        parsed_args.portion_covered,
        parsed_args.pvt_data_file,
        parsed_args.x_resolution,
        parsed_args.y_resolution,
    )
    logger.debug(
        "PV-T panel elements:\n  %s",
        "\n  ".join(
            [
                f"{element_coordinates}: {element}"
                for element_coordinates, element in pvt_panel.elements.items()
            ]
        ),
    )

    # Check that the output file is specified, and that it doesn't already exist.
    if parsed_args.output is None or parsed_args.output == "":
        logger.error(
            "%sAn output filename must be provided on the command-line interface.%s",
            BColours.FAIL,
            BColours.ENDC,
        )
        raise MissingParametersError(
            "Command-Line Interface", "An output file name must be provided."
        )
    if parsed_args.output.endswith(".yaml") or parsed_args.output.endswith(".json"):
        logger.error(
            "%sThe output filename must be irrespective of data type..%s",
            BColours.FAIL,
            BColours.ENDC,
        )
        raise Exception(
            "The output file must be irrespecitve of file extension/data type."
        )
    if os.path.isfile(f"{parsed_args.output}.yaml"):
        logger.info("The output YAML file specified already exists. Moving...")
        os.rename(f"{parsed_args.output}.yaml", f"{parsed_args.output}.yaml.1")
        logger.info("Output file successfully moved.")
    if os.path.isfile(f"{parsed_args.output}.json"):
        logger.info("The output YAML file specified already exists. Moving...")
        os.rename(f"{parsed_args.output}.json", f"{parsed_args.output}.json.1")
        logger.info("Output file successfully moved.")

    # Instantiate a hot-water tank instance based on the data.
    if parsed_args.decoupled:
        hot_water_tank: Optional[tank.Tank] = None
    else:
        hot_water_tank = hot_water_tank_from_path(parsed_args.tank_data_file)

    logger.debug(
        "PVT system information successfully parsed:\n%s\n%s",
        str(pvt_panel),
        str(hot_water_tank) if hot_water_tank is not None else "NO HOT-WATER TANK",
    )

    # Determine the Fourier number for the PV-T panel.
    logger.info("Beginning Fourier number calculation.")
    _determine_fourier_numbers(hot_water_tank, logger, parsed_args, pvt_panel)

    # Determine the operating mode of the system.
    operating_mode = OperatingMode(not parsed_args.decoupled, parsed_args.dynamic)

    if operating_mode.dynamic and operating_mode.coupled:
        logger.info(
            "Running a dynamic and coupled system.",
        )
        print(
            "{}Running a dynamic and coupled system.{}".format(
                BColours.OKGREEN,
                BColours.ENDC,
            )
        )

        # Iterate to determine the initial conditions for the run.
        logger.info("Determining consistent initial conditions at coarse resolution.")
        print(
            "Determining consistent initial conditions via successive runs at "
            f"{COARSE_RUN_RESOLUTION}s resolution."
        )

        initial_system_temperature_vector, _ = _determine_consistent_conditions(
            pvt_panel.absorber.number_of_pipes,
            layers,
            logger,
            operating_mode,
            parsed_args,
            pvt_panel,
        )
        logger.info("Consistent initial conditions determined at coarse resolution.")
        logger.info(
            "Running at the fine CLI resolution of %ss.", parsed_args.resolution
        )
        print(
            "Rough initial conditions determined at coarse resolution, refining via "
            f"successive runs at CLI resolution of {parsed_args.resolution}s."
        )
        initial_system_temperature_vector, _ = _determine_consistent_conditions(
            pvt_panel.absorber.number_of_pipes,
            layers,
            logger,
            operating_mode,
            parsed_args,
            pvt_panel,
            resolution=parsed_args.resolution,
            running_system_temperature_vector=initial_system_temperature_vector,
        )
        if initial_system_temperature_vector is None:
            raise ProgrammerJudgementFault(
                "{}The initial system conditions were not returned as ".format(
                    BColours.FAIL
                )
                + "expected when performing a coupled run.{}".format(BColours.ENDC)
            )
        logger.info(
            "Initial system temperatures successfully determined to %sK precision.",
            INITIAL_CONDITION_PRECISION,
        )
        print(
            f"Initial conditions determined to {INITIAL_CONDITION_PRECISION}K precision."
        )

        logger.info(
            "Running the model at the CLI resolution of %ss.", parsed_args.resolution
        )
        print(
            f"Running the model at the high CLI resolution of {parsed_args.resolution}s."
        )
        # Run the model at this higher resolution.
        _, system_data = pvt_system_model_main(
            parsed_args.average_irradiance,
            parsed_args.cloud_efficacy_factor,
            parsed_args.disable_logging,
            parsed_args.exchanger_data_file,
            parsed_args.initial_month,
            initial_system_temperature_vector,
            layers,
            parsed_args.location,
            operating_mode,
            parsed_args.portion_covered,
            parsed_args.pvt_data_file,
            parsed_args.resolution,
            not parsed_args.skip_2d_output,
            parsed_args.tank_data_file,
            parsed_args.use_pvgis,
            parsed_args.verbose,
            parsed_args.x_resolution,
            parsed_args.y_resolution,
            days=parsed_args.days,
            minutes=parsed_args.minutes,
            months=parsed_args.months,
            override_ambient_temperature=parsed_args.ambient_temperature,
            override_collector_input_temperature=parsed_args.collector_input_temperature,
            override_irradiance=parsed_args.solar_irradiance,
            override_mass_flow_rate=parsed_args.mass_flow_rate,
            override_wind_speed=parsed_args.wind_speed,
            run_number=1,
            start_time=parsed_args.start_time,
        )

    elif operating_mode.decoupled and operating_mode.dynamic:
        logger.info(
            "Running a dynamic and decoupled system.",
        )
        print(
            "{}Running a dynamic and decoupled system.{}".format(
                BColours.OKGREEN,
                BColours.ENDC,
            )
        )

        logger.info(
            "Running the model at the CLI resolution of %ss.", parsed_args.resolution
        )
        print(
            f"Running the model at the high CLI resolution of {parsed_args.resolution}s."
        )
        # Run the model at this higher resolution.
        _, system_data = pvt_system_model_main(
            parsed_args.average_irradiance,
            parsed_args.cloud_efficacy_factor,
            parsed_args.disable_logging,
            parsed_args.exchanger_data_file,
            parsed_args.initial_month,
            [DEFAULT_SYSTEM_TEMPERATURE]
            * (index_handler.num_temperatures(pvt_panel) - 3),
            layers,
            parsed_args.location,
            operating_mode,
            parsed_args.portion_covered,
            parsed_args.pvt_data_file,
            parsed_args.resolution,
            not parsed_args.skip_2d_output,
            parsed_args.tank_data_file,
            parsed_args.use_pvgis,
            parsed_args.verbose,
            parsed_args.x_resolution,
            parsed_args.y_resolution,
            days=parsed_args.days,
            minutes=parsed_args.minutes,
            months=parsed_args.months,
            override_ambient_temperature=parsed_args.ambient_temperature,
            override_collector_input_temperature=parsed_args.collector_input_temperature,
            override_irradiance=parsed_args.solar_irradiance,
            override_mass_flow_rate=parsed_args.mass_flow_rate,
            override_wind_speed=parsed_args.wind_speed,
            run_number=1,
            start_time=parsed_args.start_time,
        )

    elif operating_mode.decoupled and operating_mode.steady_state:
        logger.info("Running a steady-state and decoupled system.")
        print(
            f"{BColours.OKGREEN}"
            "Running a steady-state and decoupled system."
            f"{BColours.ENDC}"
        )
        # Set up a holder for information about the system.
        system_data = dict()

        if parsed_args.steady_state_data_file is not None:
            # If specified, parse the steady-state data file.
            steady_state_runs = read_yaml(parsed_args.steady_state_data_file)

            for steady_state_run in steady_state_runs:
                if parsed_args.ambient_temperature is not None:
                    steady_state_run[
                        "ambient_temperature"
                    ] = parsed_args.ambient_temperature
                if parsed_args.collector_input_temperature is not None:
                    steady_state_run[
                        "collector_input_temperature"
                    ] = parsed_args.collector_input_temperature
                if parsed_args.solar_irradiance is not None:
                    steady_state_run["irradiance"] = parsed_args.solar_irradiance
                if parsed_args.wind_speed is not None:
                    steady_state_run["wind_speed"] = parsed_args.wind_speed

            logger.info(
                "%s runs will be attempted based on the input data file.",
                len(steady_state_runs),
            )
            print(
                f"{BColours.OKGREEN}"
                f"{len(steady_state_runs)} runs will be attempted based on the "
                f"information in {parsed_args.steady_state_data_file}."
                f"{BColours.ENDC}"
            )

            with Pool(8) as worker_pool:
                logger.info(
                    "A multi-process worker pool will be used to parallelise the task."
                )
                multi_processing_output = worker_pool.map(
                    partial(
                        _multiprocessing_determine_consistent_conditions,
                        number_of_pipes=pvt_panel.absorber.number_of_pipes,
                        layers=layers,
                        logger=logger,
                        operating_mode=operating_mode,
                        parsed_args=parsed_args,
                        pvt_panel=pvt_panel,
                    ),
                    steady_state_runs,
                )
                logger.info("Multi-process worker pool successfully completed.")

            for multi_processing_output_entry in multi_processing_output:
                for key, value in multi_processing_output_entry.items():
                    system_data[key] = value

        else:
            raise ProgrammerJudgementFault(
                "Steady-state data must be specified with a JSON data file."
            )
    else:
        raise ProgrammerJudgementFault(
            "The system needs to either be run in steady-state or dynamic modes, with "
            "either `--steady-state` or `--dynamic` specified on the CLI."
        )

    # Save the data ouputted by the model.
    logger.info("Saving output data to: %s.json.", parsed_args.output)
    _save_data(FileType.JSON, logger, operating_mode, parsed_args.output, system_data)
    print(f"Model output successfully saved to {parsed_args.output}.json.")

    # If in verbose mode, output average, min, and max temperatures.
    if parsed_args.verbose:
        _output_temperature_info(logger, parsed_args, system_data)

    if parsed_args.skip_analysis:
        logger.info("Analysis will be skippted.")
        print("Skipping analysis. This can be run manually later.")
    else:
        # Conduct analysis of the data.
        logger.info("Conducting analysis.")
        print("Conducting analysis.")
        analysis.analyse(f"{parsed_args.output}.json")  # type: ignore
        print("Analysis complete. Figures can be found in `./figures`.")
        logger.info("Analysis complete.")

    logger.info("Exiting.")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except DivergentSolutionError:
        print(
            "A divergent solution occurred - have you considered the difference "
            "between Celcius and Kelvin in all your units, especially override CLI "
            "units. Consider checking this before investigating further."
        )
    except Exception:
        print("An exception occured. See /logs for details.")
        raise
