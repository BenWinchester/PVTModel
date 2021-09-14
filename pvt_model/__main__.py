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

import os
import sys

from argparse import Namespace
from functools import partial
from logging import Logger
from multiprocessing import Pool
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from . import argparser

from .__utils__ import (
    BColours,
    COARSE_RUN_RESOLUTION,
    FileType,
    SteadyStateRun,
    fourier_number,
    get_logger,
    INITIAL_CONDITION_PRECISION,
    LOGGER_NAME,
    OperatingMode,
    ProgrammerJudgementFault,
    read_yaml,
    save_data,
    SystemData,
    TemperatureName,
)

from .analysis import analysis

from .pvt_system import index_handler, tank
from .pvt_system.pvt_collector import pvt
from .pvt_system.__main__ import main as pvt_system_main
from .pvt_system.__utils__ import DivergentSolutionError

from .pvt_system.constants import (
    DEFAULT_SYSTEM_TEMPERATURE,
    DENSITY_OF_WATER,
    HEAT_CAPACITY_OF_WATER,
    THERMAL_CONDUCTIVITY_OF_WATER,
    ZERO_CELCIUS_OFFSET,
)
from .pvt_system.process_pvt_system_data import (
    hot_water_tank_from_path,
    pvt_collector_from_path,
)


# Done message:
#   The message to display when a task was successful.
DONE: str = "[   DONE   ]"

# Failed message:
#   The message to display when a task has failed.
FAILED: str = "[  FAILED  ]"

# PVT header string:
#   Text to display when instantiating the PV-T model.
PVT_HEADER_STRING = """

                                       ,
                                       ,
                          ,           ,,,           ,
                           ,,         ,,,         ,,
                            ,,,.     .,,,,      ,,,
                             ,,,,.            ,,,,.
                 ,,.              ,,,,,,,,,,,    ,         .,,.
                    ,,,,,,.   ,,,,,,,,,,,,,,,,,,,   .,,,,,,.
                       ,,,  ,,,,,,,,,,,,,,,,,,,,,,,  ,,,.
                           ,,,,,,,,,,,,,,,,,,,,,,,,,
                          ,,,,,,,,,,,,,,,,,,,,,,,,,,,
           .,,,,,,,,,,,,  ,,,,,,,,,.,.,,,,,,,,,,,,,,,  ,,,,,,,,,,,,,

               _    _ ______       _______ _____                 _
              | |  | |  ____|   /\\|__   __|  __ \\               | |
              | |__| | |__     /  \\  | |  | |__) |_ _ _ __   ___| |
              |  __  |  __|   / /\\ \\ | |  |  ___/ _` | '_ \\ / _ \\ |
              | |  | | |____ / ____ \\| |  | |  | (_| | | | |  __/ |
              |_|  |_|______/_/    \\_\\_|  |_|   \\__,_|_| |_|\\___|_|


                    Hybrid Electric And Thermal Panel Model
                      Copyright Benedict Winchester, 2021

              For more information, contact Benedict Winchester at
                         benedict.winchester@gmail.com

"""


def _get_system_fourier_numbers(
    hot_water_tank: Optional[tank.Tank], pvt_collector: pvt.PVT, resolution: float
) -> Dict[TemperatureName, float]:
    """
    Determine the Fourier numbers of the various system components.

    :param hot_water_tank:
        A :class:`tank.Tank` instance representing the hot-water tank in the system.

    :param pvt_collector:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The resolution being used for the model.

    :return:
        The Fourier coefficients for the various components being modelled.

    """

    # Determine the Fourier coefficients of the panel's layers.
    fourier_number_map: Dict[TemperatureName, float] = dict()
    if pvt_collector.glass is not None:
        fourier_number_map[TemperatureName.glass] = round(
            fourier_number(
                pvt_collector.glass.thickness,
                pvt_collector.glass.conductivity,
                pvt_collector.glass.density,
                pvt_collector.glass.heat_capacity,
                resolution,
            ),
            2,
        )
    if pvt_collector.pv is not None:
        fourier_number_map[TemperatureName.pv] = round(
            fourier_number(
                pvt_collector.pv.thickness,
                pvt_collector.pv.conductivity,
                pvt_collector.pv.density,
                pvt_collector.pv.heat_capacity,
                resolution,
            ),
            2,
        )
    fourier_number_map[TemperatureName.absorber] = round(
        fourier_number(
            pvt_collector.absorber.thickness,
            pvt_collector.absorber.conductivity,
            pvt_collector.absorber.density,
            pvt_collector.absorber.heat_capacity,
            resolution,
        ),
        2,
    )
    fourier_number_map[TemperatureName.htf] = round(
        fourier_number(
            pvt_collector.absorber.inner_pipe_diameter,
            THERMAL_CONDUCTIVITY_OF_WATER,
            DENSITY_OF_WATER,
            pvt_collector.absorber.htf_heat_capacity,
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


def _calculate_and_print_fourier_numbers(
    hot_water_tank: Optional[tank.Tank],
    logger: Logger,
    parsed_args: Namespace,
    pvt_collector: pvt.PVT,
) -> None:
    """
    Determines, prints, and logs the various Fourier numbers.

    :param hot_water_tank:
        A :class:`tank.Tank` instance representing the hot-water tank being modelled.

    :param logger:
        The logger being used for the run.

    :param parsed_args:
        The parsed commnand-line arguments.

    :param pvt_collector:
        The pvt panel, represented as a :class:`pvt_collector.pvt.PVT` instance.

    """

    # Determine the Fourier numbers.
    fourier_number_map = _get_system_fourier_numbers(
        hot_water_tank, pvt_collector, parsed_args.resolution
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
    pvt_collector: pvt.PVT,
    *,
    override_ambient_temperature: Optional[float] = None,
    override_collector_input_temperature: Optional[float] = None,
    override_irradiance: Optional[float] = None,
    override_mass_flow_rate: Optional[float] = None,
    override_wind_speed: Optional[float] = None,
    resolution: int = COARSE_RUN_RESOLUTION,
    run_depth: int = 1,
    running_system_temperature_vector: Optional[List[float]] = None,
) -> Tuple[Optional[List[float]], Dict[float, SystemData]]:
    """
    Determines the initial system temperatures for the run.

    This function is called recursively until the temperature vector outputted by the
    system is consistent within some error given by the
    :module:`INITIAL_CONDITION_PRECISION` variable.

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

    :param pvt_collector:
        The :class:`pvt.PVT` instance representing the pvt panel being modelled.

    :param override_ambient_tempearture:
        If specified, this can be used as a value to override the weather forecaster's
        inbuilt ambient temperature.

    :param override_mass_flow_rate:
        If specified, this can be used as a value to override the collector input file's
        default mass-flow rate.

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
            ] * index_handler.num_temperatures(pvt_collector)
        else:
            running_system_temperature_vector = [DEFAULT_SYSTEM_TEMPERATURE] * (
                index_handler.num_temperatures(pvt_collector) - 3
            )

    # Call the model to generate the output of the run.
    logger.info("Running the model. Run number %s.", run_depth)
    final_temperature_vector, system_data = pvt_system_main(
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
        override_mass_flow_rate=override_mass_flow_rate
        if override_mass_flow_rate is not None
        else parsed_args.mass_flow_rate,
        override_wind_speed=override_wind_speed,
        run_number=run_depth,
        start_time=parsed_args.start_time,
    )

    # If in verbose mode, output average, min, and max temperatures.
    if parsed_args.verbose:
        _determine_and_print_average_temperatures(logger, parsed_args, system_data)

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
        pvt_collector,
        override_ambient_temperature=override_ambient_temperature,
        override_collector_input_temperature=override_collector_input_temperature,
        override_irradiance=override_irradiance,
        override_mass_flow_rate=override_mass_flow_rate
        if override_mass_flow_rate is not None
        else parsed_args.mass_flow_rate,
        override_wind_speed=override_wind_speed,
        resolution=resolution,
        run_depth=run_depth + 1,
        running_system_temperature_vector=final_temperature_vector.tolist(),
    )


def _multiprocessing_determine_consistent_conditions(
    steady_state_run: SteadyStateRun,
    *,
    number_of_pipes: int,
    layers: Set[TemperatureName],
    logger: Logger,
    operating_mode: OperatingMode,
    parsed_args: Namespace,
    pvt_collector: pvt.PVT,
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

    :param pvt_collector:
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
        pvt_collector,
        override_ambient_temperature=steady_state_run.ambient_temperature,
        override_collector_input_temperature=steady_state_run.collector_input_temperature
        + ZERO_CELCIUS_OFFSET,
        override_irradiance=steady_state_run.irradiance,
        override_mass_flow_rate=steady_state_run.mass_flow_rate,
        override_wind_speed=steady_state_run.wind_speed,
    )

    return system_data_entry


def _determine_and_print_average_temperatures(
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


def main(args) -> None:  # pylint: disable=too-many-branches
    """
    The main module for the code.

    :param args:
        The command-line arguments passed into the component.

    """

    # Print the header string.
    print(PVT_HEADER_STRING)

    # Parse the arguments passed in.
    parsed_args = argparser.parse_args(args)

    # Initialise logging.
    logger = get_logger(parsed_args.disable_logging, LOGGER_NAME, parsed_args.verbose)
    logger.info(
        "%s PVT model instantiated. %s\nCommand: %s", "=" * 20, "=" * 20, " ".join(args)
    )
    print(
        "HEAT model of a hybrid PV-T collector instantiated{}.".format(
            f"{BColours.OKTEAL} in verbose mode{BColours.ENDC}"
            if parsed_args.verbose
            else ""
        )
    )

    print(
        "Verifying input information and arguments {}    ".format(
            "." * 21,
        ),
        end="",
    )
    try:
        # Validate the CLI arguments.
        layers = argparser.check_args(
            parsed_args,
            logger,
            read_yaml(parsed_args.pvt_data_file)["absorber"]["number_of_pipes"],
        )
    except argparser.ArgumentMismatchError:
        print(FAILED)
        raise
    print(DONE)

    # Parse the PVT system information and generate a PVT panel based on the args for
    # use in Fourier-number calculations.
    pvt_collector = pvt_collector_from_path(
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
                for element_coordinates, element in pvt_collector.elements.items()
            ]
        ),
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
        str(pvt_collector),
        str(hot_water_tank) if hot_water_tank is not None else "NO HOT-WATER TANK",
    )

    # Determine the Fourier number for the PV-T panel.
    logger.info("Beginning Fourier number calculation.")
    _calculate_and_print_fourier_numbers(
        hot_water_tank, logger, parsed_args, pvt_collector
    )

    # Determine the operating mode of the system.
    operating_mode = OperatingMode(not parsed_args.decoupled, parsed_args.dynamic)

    if operating_mode.dynamic:
        logger.info(
            "Running a dynamic and %scoupled system.",
            "de" if operating_mode.decoupled else "",
        )
        print(
            "{}Running a dynamic and {}coupled system.{}".format(
                BColours.OKGREEN,
                "de" if operating_mode.decoupled else "",
                BColours.ENDC,
            )
        )

        if operating_mode.coupled:
            # The initial system conditions for the run need to be determined so that
            # they match up at the start and end of the time period being modelled,
            # i.e., so that there is no transient in the data.

            # Iterate to determine the initial conditions for the run at a rough
            # resolution.
            logger.info(
                "Determining consistent initial conditions at coarse resolution."
            )
            print(
                "Determining consistent initial conditions via successive runs at "
                f"{COARSE_RUN_RESOLUTION}s resolution."
            )

            initial_system_temperature_vector, _ = _determine_consistent_conditions(
                pvt_collector.absorber.number_of_pipes,
                layers,
                logger,
                operating_mode,
                parsed_args,
                pvt_collector,
            )
            logger.info(
                "Consistent initial conditions determined at coarse resolution."
            )

            # Run again at a fine resolution to better determine the initial conditions.
            print(
                "Rough initial conditions determined at coarse resolution, refining via "
                f"successive runs at CLI resolution of {parsed_args.resolution}s."
            )
            initial_system_temperature_vector, _ = _determine_consistent_conditions(
                pvt_collector.absorber.number_of_pipes,
                layers,
                logger,
                operating_mode,
                parsed_args,
                pvt_collector,
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

        else:
            # In the case of a decoupled and dynamic model, usually a step change is
            # being modelled. In these instances, the system is run for some time,
            # determined by the data fed in, and the initial conditions can simply be
            # taken as the default conditions.
            initial_system_temperature_vector = [DEFAULT_SYSTEM_TEMPERATURE] * (
                index_handler.num_temperatures(pvt_collector) - 3
            )

        logger.info(
            "Running the model at the CLI resolution of %ss.", parsed_args.resolution
        )
        print(
            f"Running the model at the high CLI resolution of {parsed_args.resolution}s."
        )
        # Run the model at this higher resolution.
        _, system_data = pvt_system_main(
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
            steady_state_runs = [
                SteadyStateRun.from_data(entry)
                for entry in read_yaml(parsed_args.steady_state_data_file)
            ]

            for steady_state_run in steady_state_runs:
                if parsed_args.ambient_temperature is not None:
                    steady_state_run.ambient_temperature = (
                        parsed_args.ambient_temperature
                    )
                if parsed_args.collector_input_temperature is not None:
                    steady_state_run.collector_input_temperature = (
                        parsed_args.collector_input_temperature
                    )
                if parsed_args.solar_irradiance is not None:
                    steady_state_run.irradiance = parsed_args.solar_irradiance
                if parsed_args.wind_speed is not None:
                    steady_state_run.wind_speed = parsed_args.wind_speed

            logger.info(
                "%s run%s will be attempted based on the input data file.",
                len(steady_state_runs),
                "s" if len(steady_state_runs) > 1 else "",
            )
            print(
                f"{BColours.OKGREEN}"
                + "{} run{} will be attempted based on the ".format(
                    len(steady_state_runs), "s" if len(steady_state_runs) > 1 else ""
                )
                + f"information in {parsed_args.steady_state_data_file}."
                + f"{BColours.ENDC}"
            )

            # with Pool(8) as worker_pool:
            #     logger.info(
            #         "A multi-process worker pool will be used to parallelise the task."
            #     )
            #     multi_processing_output = worker_pool.map(
            #         partial(
            #             _multiprocessing_determine_consistent_conditions,
            #             number_of_pipes=pvt_collector.absorber.number_of_pipes,
            #             layers=layers,
            #             logger=logger,
            #             operating_mode=operating_mode,
            #             parsed_args=parsed_args,
            #             pvt_collector=pvt_collector,
            #         ),
            #         steady_state_runs
            #     )
            #     logger.info("Multi-process worker pool successfully completed.")

            for run_number, steady_state_run in enumerate(
                tqdm(steady_state_runs, desc="steady state runs", unit="run"), 1
            ):
                logger.info(
                    "Carrying out steady-state run %s of %s.",
                    run_number,
                    len(steady_state_runs),
                )
                try:
                    output = _multiprocessing_determine_consistent_conditions(
                        steady_state_run,
                        number_of_pipes=pvt_collector.absorber.number_of_pipes,
                        layers=layers,
                        logger=logger,
                        operating_mode=operating_mode,
                        parsed_args=parsed_args,
                        pvt_collector=pvt_collector,
                    )
                except RecursionError as e:
                    logger.error(
                        "Recursion error processing steady state run.\nRun: %s\nMsg: %s",
                        steady_state_run,
                        str(e),
                    )
                    continue
                except DivergentSolutionError as e:
                    logger.error(
                        "A divergent solution occurred - have you considered the "
                        "difference between Celcius and Kelvin in all your units, "
                        "especially override CLI units. Consider checking this before "
                        "investigating further.\nRun attempted: %s\nError: %s",
                        steady_state_run,
                        str(e)
                    )
                    continue

                for key, value in output.items():
                    system_data[f"run_{run_number}_T_in_{key}degC"] = value

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
    save_data(FileType.JSON, logger, operating_mode, parsed_args.output, system_data)
    print(f"Model output successfully saved to {parsed_args.output}.json.")

    # If in verbose mode, output average, min, and max temperatures.
    if parsed_args.verbose:
        _determine_and_print_average_temperatures(logger, parsed_args, system_data)

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
