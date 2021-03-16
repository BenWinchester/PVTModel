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
from logging import Logger
from statistics import mean
from typing import Any, Dict, List, Optional

import json
import yaml

from . import argparser

from .__utils__ import (
    CarbonEmissions,
    COARSE_RUN_RESOLUTION,
    FileType,
    fourier_number,
    get_logger,
    INITIAL_CONDITION_PRECISION,
    MissingParametersError,
    LOGGER_NAME,
    SystemData,
    TemperatureName,
    TotalPowerData,
)

from .analysis import analysis

from .pvt_system_model import index_handler, tank
from .pvt_system_model.pvt_panel import pvt
from .pvt_system_model.__main__ import main as pvt_system_model_main

from .pvt_system_model.constants import (
    DEFAULT_SYSTEM_TEMPERATURE,
    DENSITY_OF_WATER,
    HEAT_CAPACITY_OF_WATER,
    THERMAL_CONDUCTIVITY_OF_WATER,
)
from .pvt_system_model.process_pvt_system_data import (
    hot_water_tank_from_path,
    pvt_panel_from_path,
)


def _get_system_fourier_numbers(
    hot_water_tank: tank.Tank, pvt_panel: pvt.PVT, resolution: float
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
    fourier_number_map[TemperatureName.collector] = round(
        fourier_number(
            pvt_panel.collector.thickness,
            pvt_panel.collector.conductivity,
            pvt_panel.collector.density,
            pvt_panel.collector.heat_capacity,
            resolution,
        ),
        2,
    )
    fourier_number_map[TemperatureName.htf] = round(
        fourier_number(
            pvt_panel.collector.inner_pipe_diameter,
            THERMAL_CONDUCTIVITY_OF_WATER,
            DENSITY_OF_WATER,
            pvt_panel.collector.htf_heat_capacity,
            resolution,
        ),
        2,
    )
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
    hot_water_tank: tank.Tank,
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


def _determine_initial_conditions(
    number_of_pipes: int,
    logger: Logger,
    parsed_args: Namespace,
    resolution: int = COARSE_RUN_RESOLUTION,
    run_depth: int = 1,
    running_system_temperature_vector: Optional[List[float]] = None,
) -> List[float]:
    """
    Determines the initial system temperatures for the run.

    :param number_of_pipes:
        The number of pipes on the base of the hot-water collector.

    :param logger:
        The logger for the run.

    :param parsed_args:
        The parsed command-line arguments.

    :param resolution:
        The resolution for the run, measured in seconds.

    :param run_depth:
        The depth of the recursion.

    :param running_system_temperature_vector:
        The current vector for the system temperatures, used to commence a run and
        compare with the previous run.

    :return:
        The initial system temperature which fits within the desired resolution.

    """

    # Fetch the initial temperature conditions if not passed in:
    if running_system_temperature_vector is None:
        running_system_temperature_vector = [
            DEFAULT_SYSTEM_TEMPERATURE
        ] * index_handler.num_temperatures(
            number_of_pipes,
            (parsed_args.x_resolution),
            (parsed_args.y_resolution),
        )

    # Call the model to generate the output of the run.
    logger.info("Running the model. Run number %s.", run_depth)
    final_temperature_vector, _ = pvt_system_model_main(
        parsed_args.average_irradiance,
        parsed_args.cloud_efficacy_factor,
        parsed_args.days,
        parsed_args.exchanger_data_file,
        parsed_args.initial_month,
        running_system_temperature_vector,
        parsed_args.location,
        parsed_args.months,
        parsed_args.portion_covered,
        parsed_args.pvt_data_file,
        resolution,
        run_depth,
        not parsed_args.skip_2d_output,
        parsed_args.start_time,
        parsed_args.tank_data_file,
        # parsed_args.unglazed,
        parsed_args.use_pvgis,
        parsed_args.verbose,
        parsed_args.x_resolution,
        parsed_args.y_resolution,
    )

    # If all the temperatures are within the desired limit, return the temperatures.
    if all(
        abs(final_temperature_vector - running_system_temperature_vector)
        <= INITIAL_CONDITION_PRECISION
    ):
        logger.info(
            "Initial temperatures consistent. Max difference: %sK",
            max(abs(final_temperature_vector - running_system_temperature_vector)),
        )
        return final_temperature_vector.tolist()

    logger.info(
        "Initial temperatures not consistent. Max difference: %sK",
        max(abs(final_temperature_vector - running_system_temperature_vector)),
    )
    # Otherwise, call the method recursively.
    return _determine_initial_conditions(
        number_of_pipes,
        logger,
        parsed_args,
        resolution,
        run_depth + 1,
        final_temperature_vector.tolist(),
    )


def _print_temperature_info(
    average_temperature_map: Dict[str, float],
    logger: Logger,
    maximum_temperature_map: Dict[str, float],
    minimum_temperature_map: Dict[str, float],
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
        "Average temperatures for the run in degC:\n{}\n{}".format(
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
        "Maximum temperatures for the run in degC:\n{}\n{}".format(
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
        "Minimum temperatures for the run in degC:\n{}\n{}".format(
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
    )


def _save_data(
    file_type: FileType,
    output_file_name: str,
    system_data: Dict[int, SystemData],
    carbon_emissions: Optional[CarbonEmissions] = None,
    total_power_data: Optional[TotalPowerData] = None,
) -> None:
    """
    Save data when called. The data entry should be appended to the file.
    :param file_type:
        The file type that's being saved.
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
    system_data_dict: Dict[int, Dict[str, Any]] = {
        key: dataclasses.asdict(value) for key, value in system_data.items()
    }

    # If we're saving YAML data part-way through, then append to the file.
    if file_type == FileType.YAML:
        with open(f"{output_file_name}.yaml", "a") as output_yaml_file:
            yaml.dump(
                system_data_dict,
                output_yaml_file,
            )

    # If we're dumping JSON, open the file, and append to it.
    if file_type == FileType.JSON:
        # Append the total power and emissions data for the run.
        if total_power_data is not None:
            system_data_dict.update(dataclasses.asdict(total_power_data))  # type: ignore
        if carbon_emissions is not None:
            system_data_dict.update(dataclasses.asdict(carbon_emissions))  # type: ignore

        # Save the data
        # If this is the initial dump, then create the file.
        if not os.path.isfile(f"{output_file_name}.json"):
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


def main(args) -> None:
    """
    The main module for the code.

    :param args:
        The command-line arguments passed into the component.

    """

    # Parse the arguments passed in.
    parsed_args = argparser.parse_args(args)

    # Initialise logging.
    logger = get_logger(LOGGER_NAME, parsed_args.verbose)
    logger.info(
        "%s PVT model instantiated. %s\nCommand: %s", "=" * 20, "=" * 20, " ".join(args)
    )
    print("PVT model instantiated.")

    # Check that the output file is specified, and that it doesn't already exist.
    if parsed_args.output is None or parsed_args.output == "":
        logger.error(
            "An output filename must be provided on the command-line interface."
        )
        raise MissingParametersError(
            "Command-Line Interface", "An output file name must be provided."
        )
    if parsed_args.output.endswith(".yaml") or parsed_args.output.endswith(".json"):
        logger.error("The output filename must be irrespective of data type..")
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

    # Parse the PVT system information and generate a PVT panel based on the args.
    pvt_panel = pvt_panel_from_path(
        parsed_args.portion_covered,
        parsed_args.pvt_data_file,
        parsed_args.x_resolution,
        parsed_args.y_resolution,
    )
    # Instantiate a hot-water tank instance based on the data.
    hot_water_tank = hot_water_tank_from_path(parsed_args.tank_data_file)
    logger.debug(
        "PVT system information successfully parsed:\n%s\n%s",
        str(pvt_panel),
        str(hot_water_tank),
    )

    # Determine the Fourier number for the PV-T panel.
    logger.info("Beginning Fourier number calculation.")
    _determine_fourier_numbers(hot_water_tank, logger, parsed_args, pvt_panel)

    # Iterate to determine the initial conditions for the run.
    logger.info("Determining consistent initial conditions at coarse resolution.")
    print(
        "Determining consistent initial conditions via successive runs at "
        f"{COARSE_RUN_RESOLUTION}s resolution."
    )
    initial_system_temperature_vector = _determine_initial_conditions(
        pvt_panel.collector.number_of_pipes, logger, parsed_args
    )
    print(
        "Rough initial conditions determined at coarse resolution, refining via "
        f"successive runs at CLI resolution of {parsed_args.resolution}s."
    )
    initial_system_temperature_vector = _determine_initial_conditions(
        pvt_panel.collector.number_of_pipes,
        logger,
        parsed_args,
        resolution=parsed_args.resolution,
        running_system_temperature_vector=initial_system_temperature_vector,
    )
    logger.info(
        "Initial system temperatures successfully determined to %sK precision.",
        INITIAL_CONDITION_PRECISION,
    )
    print(f"Initial conditions determined to {INITIAL_CONDITION_PRECISION}K precision.")

    logger.info(
        "Running the model at the CLI resolution of %ss.", parsed_args.resolution
    )
    print(f"Running the model at the high CLI resolution of {parsed_args.resolution}s.")
    # Run the model at this higher resolution.
    _, system_data = pvt_system_model_main(
        parsed_args.average_irradiance,
        parsed_args.cloud_efficacy_factor,
        parsed_args.days,
        parsed_args.exchanger_data_file,
        parsed_args.initial_month,
        initial_system_temperature_vector,
        parsed_args.location,
        parsed_args.months,
        parsed_args.portion_covered,
        parsed_args.pvt_data_file,
        parsed_args.resolution,
        1,
        not parsed_args.skip_2d_output,
        parsed_args.start_time,
        parsed_args.tank_data_file,
        # parsed_args.unglazed,
        parsed_args.use_pvgis,
        parsed_args.verbose,
        parsed_args.x_resolution,
        parsed_args.y_resolution,
    )

    # Save the data ouputted by the model.
    logger.info("Saving output data to: %s.json.", parsed_args.output)
    _save_data(FileType.JSON, parsed_args.output, system_data)
    print(f"Model output successfully saved to {parsed_args.output}.json.")

    # If in verbose mode, output average, min, and max temperatures.
    if parsed_args.verbose:
        # Determine the average, minimum, and maximum temperatures.
        average_temperature_map = {
            "glass": round(
                mean({entry.glass_temperature for entry in system_data.values()}), 3
            ),
            "pv": round(
                mean({entry.pv_temperature for entry in system_data.values()}), 3
            ),
            "absorber": round(
                mean({entry.collector_temperature for entry in system_data.values()}), 3
            ),
            "htf": round(
                mean({entry.bulk_water_temperature for entry in system_data.values()}),
                3,
            ),
            "tank": round(
                mean({entry.tank_temperature for entry in system_data.values()}), 3
            ),
        }
        maximum_temperature_map = {
            "glass": max(
                {round(entry.glass_temperature, 3) for entry in system_data.values()}
            ),
            "pv": max(
                {round(entry.pv_temperature, 3) for entry in system_data.values()}
            ),
            "absorber": max(
                {
                    round(entry.collector_temperature, 3)
                    for entry in system_data.values()
                }
            ),
            "htf": max(
                {
                    round(entry.bulk_water_temperature, 3)
                    for entry in system_data.values()
                }
            ),
            "tank": max(
                {round(entry.tank_temperature, 3) for entry in system_data.values()}
            ),
        }
        minimum_temperature_map = {
            "glass": min(
                {round(entry.glass_temperature, 3) for entry in system_data.values()}
            ),
            "pv": min(
                {round(entry.pv_temperature, 3) for entry in system_data.values()}
            ),
            "absorber": min(
                {
                    round(entry.collector_temperature, 3)
                    for entry in system_data.values()
                }
            ),
            "htf": min(
                {
                    round(entry.bulk_water_temperature, 3)
                    for entry in system_data.values()
                }
            ),
            "tank": min(
                {round(entry.tank_temperature, 3) for entry in system_data.values()}
            ),
        }

        # Print these out to the console.
        logger.info("Average, minimum, and maximum temperatures determined.")
        _print_temperature_info(
            average_temperature_map,
            logger,
            maximum_temperature_map,
            minimum_temperature_map,
        )

    # Conduct analysis of the data.
    logger.info("Conducting analysis.")
    analysis.analyse(f"{parsed_args.output}.json")
    print("Analysis complete. Figures can be found in `./figures`.")
    logger.info("Analysis complete.")

    logger.info("Exiting.")
    return


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception:
        print("An exception occured. See /logs for details.")
        raise
