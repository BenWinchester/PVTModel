#!/usr/bin/python3.7
########################################################################################
# __main__.py - The main module for this, my first, PV-T model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The main module for the PV-T model.

This module coordinates the time-stepping of the itterative module, calling out to the
various modules where necessary, as well as reading in the various data files and
processing the command-line arguments that define the scope of the model run.

"""

import dataclasses
import datetime
import os
import pdb
import sys

from argparse import Namespace
from typing import Any, Dict, Optional, Tuple

import json
import numpy
import pytz
import yaml

from dateutil.relativedelta import relativedelta
from scipy import linalg  # type: ignore

from . import (
    argparser,
    load,
    mains_power,
    matrix,
    pipe,
    process_pvt_system_data,
    tank,
    weather,
)

from .pvt_panel import pvt

from .constants import (
    DENSITY_OF_WATER,
    INITIAL_SYSTEM_TEMPERATURE_VECTOR,
    ZERO_CELCIUS_OFFSET,
)

from .__utils__ import (  # pylint: disable=unused-import
    CarbonEmissions,
    FileType,
    get_logger,
    LOGGER_NAME,
    MissingParametersError,
    ProgrammerJudgementFault,
    SystemData,
    time_iterator,
    TotalPowerData,
)


# The temperature of hot-water required by the end-user, measured in Kelvin.
HOT_WATER_DEMAND_TEMP = 60 + ZERO_CELCIUS_OFFSET
# The initial date and time for the simultion to run from.
DEFAULT_INITIAL_DATE_AND_TIME = datetime.datetime(2005, 1, 1, 0, 0, tzinfo=pytz.UTC)
# The average temperature of the air surrounding the tank, which is internal to the
# household, measured in Kelvin.
INTERNAL_HOUSEHOLD_AMBIENT_TEMPERATURE = ZERO_CELCIUS_OFFSET + 20  # [K]
# Get the logger for the component.
logger = get_logger(LOGGER_NAME)
# Folder containing the solar irradiance profiles
SOLAR_IRRADIANCE_FOLDERNAME = "solar_irradiance_profiles"
# Folder containing the temperature profiles
TEMPERATURE_FOLDERNAME = "temperature_profiles"
# Name of the weather data file.
WEATHER_DATA_FILENAME = "weather.yaml"


def _date_and_time_from_time_step(
    initial_date_and_time: datetime.datetime,
    final_date_and_time: datetime.datetime,
    time_step: numpy.float64,
) -> datetime.datetime:
    """
    Returns a :class:`datetime.datetime` instance representing the current time.

    :param initial_date_and_time:
        The initial date and time for the model run.

    :param final_date_and_time:
        The final date and time for the model run.

    :param time_step:
        The current time step, measured in seconds from the start of the run.

    :return:
        The current date and time, based on the iterative point.

    :raises: ProgrammerJudgementFault
        Raised if the date and time being returned is greater than the maximum for the
        run.

    """

    date_and_time = initial_date_and_time + relativedelta(seconds=time_step)

    if date_and_time > final_date_and_time:
        raise ProgrammerJudgementFault(
            "The model reached a time step greater than the maximum time step."
        )

    return date_and_time


def _get_load_system(location: str) -> load.LoadSystem:
    """
    Instantiates a :class:`load.LoadSystem` instance based on the file data.

    :param locataion:
        The location being considered for which to instantiate the load system.

    :return:
        An instantiated :class:`load.LoadSystem` based on the input location.

    """

    load_profiles = {
        os.path.join(location, "load_profiles", filename)
        for filename in os.listdir(os.path.join(location, "load_profiles"))
    }
    load_system: load.LoadSystem = load.LoadSystem.from_data(load_profiles)
    return load_system


def _get_weather_forecaster(
    average_irradiance: bool, location: str, use_pvgis: bool
) -> weather.WeatherForecaster:
    """
    Instantiates a :class:`weather.WeatherForecaster` instance based on the file data.

    :param average_irradiance:
        Whether to use an average irradiance profile for the month (True) or use
        irradiance profiles for each day individually (False).

    :param location:
        The location currently being considered, and for which to instantiate the
        :class:`weather.WeatherForecaster` instance.

    :param use_pvgis:
        Whether data from the PVGIS (Photovoltaic Geographic Information Survey) should
        be used (True) or not (False).

    :return:
        An instantiated :class:`weather.WeatherForecaster` instance based on the
        supplied parameters.

    """

    solar_irradiance_filenames = {
        os.path.join(location, SOLAR_IRRADIANCE_FOLDERNAME, filename)
        for filename in os.listdir(os.path.join(location, SOLAR_IRRADIANCE_FOLDERNAME))
    }
    temperature_filenames = {
        os.path.join(location, TEMPERATURE_FOLDERNAME, filename)
        for filename in os.listdir(os.path.join(location, TEMPERATURE_FOLDERNAME))
    }

    weather_forecaster: weather.WeatherForecaster = weather.WeatherForecaster.from_data(
        average_irradiance,
        os.path.join(location, WEATHER_DATA_FILENAME),
        solar_irradiance_filenames,
        temperature_filenames,
        use_pvgis,
    )
    return weather_forecaster


def _save_data(
    file_type: FileType,
    output_file_name: str,
    system_data: Dict[datetime.datetime, SystemData],
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
    system_data_dict: Dict[str, Dict[str, Any]] = {
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
            system_data_dict.update(dataclasses.asdict(total_power_data))
        if carbon_emissions is not None:
            system_data_dict.update(dataclasses.asdict(carbon_emissions))

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


# def _temperature_vector_gradient(
#     temperature_vector: Tuple[float, float, float, float, float],
#     time: numpy.float64,
#     final_date_and_time: datetime.datetime,
#     hot_water_tank: tank.Tank,
#     initial_date_and_time: datetime.datetime,
#     parsed_args: Namespace,
#     pvt_panel: pvt.PVT,
#     weather_forecaster: weather.WeatherForecaster,
# ) -> Tuple[float, float, float, float, float]:
#     """
#     Computes the vector of the temperature gradients at some time t.

#     :param temperature_vector:
#         A `tuple` representing a vector containing:
#         - the glass temperature,
#         - the pv layer temperature,
#         - the collector temperature,
#         - the bulk-water temperature,
#         - the hot-water tank tempreature.

#         All of these temperature values are measured in Kelvin.

#     :param time:
#         The current time in seconds from the beginning of the model run.

#     :param final_date_and_time:
#         The final date and time for the simulation run.

#     :param heat_exchanger:
#         The heat exchanger between the HTF and the tank.

#     :param hot_water_tank:
#         Instance representing the hot-water tank.

#     :param initial_date_and_time:
#         The initial date and time for the simulation run.

#     :param parsed_args:
#         Parsed arguments from the command line, stored as a :class:`argparse.Namespace`.

#     :param pvt_panel:
#         An instance representing the PVT panel.

#     :param weather_forecaster:
#         A :class:`weather.WeatherForecaster` containing weather information and exposing
#         functions enabling the weather conditions to be calculated at some given time.

#     :return:
#         A `tuple` representing a vector containing:
#         - the differential of the glass temperature with respect to time,
#         - the differential of the pv layer temperature with respect to time,
#         - the differential of the collector temperature with respect to time,
#         - the differential of the bulk-water temperature with respect to time,
#         - the differential of the hot-water tank temperature with respect to time.

#     """

#     try:
#         # Determine the current date and time.
#         date_and_time = _date_and_time_from_time_step(
#             initial_date_and_time, final_date_and_time, time
#         )
#     except ProgrammerJudgementFault:
#         logger.error("The system reached a timestep greater than the maximum.")
#         date_and_time = _date_and_time_from_time_step(
#             initial_date_and_time, final_date_and_time, time - 86400
#         )

#     if any([value > 350 or value < 270 for value in temperature_vector]):
#         logger.debug(
#             "Temperatures exceed the bounds specified for logging an error. "
#             "Time: %s. Temperature vector: %s",
#             date_and_time,
#             temperature_vector,
#         )

#     # Unpack the temperature tuple.
#     (
#         glass_temperature,
#         pv_temperature,
#         collector_temperature,
#         bulk_water_temperature,
#         tank_temperature,
#     ) = temperature_vector

#     # Determine the environmental conditions at the current time step.
#     weather_conditions = weather_forecaster.get_weather(
#         pvt_panel.latitude,
#         pvt_panel.longitude,
#         parsed_args.cloud_efficacy_factor,
#         date_and_time,
#     )

#     glass_temperature_gradient = panel_utils.glass_temperature_gradient(
#         collector_temperature,
#         glass_temperature,
#         pv_temperature,
#         pvt_panel,
#         weather_conditions,
#     )  # [K/s]

#     pv_temperature_gradient = panel_utils.pv_temperature_gradient(
#         collector_temperature,
#         glass_temperature,
#         pv_temperature,
#         pvt_panel,
#         weather_conditions,
#     )  # [K/s]

#     collector_temperature_gradient = panel_utils.collector_temperature_gradient(
#         bulk_water_temperature,
#         collector_temperature,
#         glass_temperature,
#         pv_temperature,
#         pvt_panel,
#         weather_conditions,
#     )  # [K/s]

#     bulk_water_temperature_gradient = panel_utils.bulk_water_temperature_gradient(
#         bulk_water_temperature, collector_temperature, pvt_panel, 0
#     )  # [K/s]

#     tank_temperature_gradient = (
#         -hot_water_tank.heat_loss(
#             weather_conditions.ambient_tank_temperature, tank_temperature
#         )  # [W]
#     ) / (
#         hot_water_tank.mass * hot_water_tank.heat_capacity  # [J/K]
#     )  # [K/s]

#     temperature_vector_gradient = (
#         glass_temperature_gradient,
#         pv_temperature_gradient,
#         collector_temperature_gradient,
#         bulk_water_temperature_gradient,
#         tank_temperature_gradient,
#     )

#     if any([value > 1 for value in temperature_vector_gradient]):
#         logger.debug(
#             "Temperature gradient vector too high. Time: %s. "
#             "Temperature gradient vector: %s.",
#             date_and_time,
#             temperature_vector_gradient,
#         )

#     return temperature_vector_gradient


def main(args) -> None:
    """
    The main module for the code.

    :param args:
        The command-line arguments passed into the component.

    """

    logger.info("Beginning run of PVT model.\nCommand: %s", " ".join(args))

    # Parse the system arguments from the commandline.
    parsed_args = argparser.parse_args(args)

    # Set up numpy printing style.
    numpy.set_printoptions(formatter={"float": "{: 0.3f}".format})

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

    # Set up the weather module.
    weather_forecaster = _get_weather_forecaster(
        parsed_args.average_irradiance, parsed_args.location, parsed_args.use_pvgis
    )
    logger.info("Weather forecaster successfully instantiated: %s", weather_forecaster)

    # Set up the load module.
    load_system = _get_load_system(parsed_args.location)
    logger.info(
        "Load system successfully instantiated: %s",
        load_system,
    )

    # Initialise the PV-T panel.
    pvt_panel = process_pvt_system_data.pvt_panel_from_path(
        INITIAL_SYSTEM_TEMPERATURE_VECTOR[3],
        parsed_args.portion_covered,
        parsed_args.pvt_data_file,
        parsed_args.unglazed,
    )
    logger.info("PV-T panel successfully instantiated: %s", pvt_panel)

    # Instantiate the rest of the PVT system.
    heat_exchanger = process_pvt_system_data.heat_exchanger_from_path(
        parsed_args.exchanger_data_file
    )
    logger.info("Heat exchanger successfully instantiated: %s", heat_exchanger)
    hot_water_tank = process_pvt_system_data.hot_water_tank_from_path(
        parsed_args.tank_data_file
    )
    logger.info("Hot-water tank successfully instantiated: %s", hot_water_tank)

    # Instantiate the two pipes used to store input and output temperature values.
    collector_to_tank_pipe = pipe.Pipe(temperature=INITIAL_SYSTEM_TEMPERATURE_VECTOR[3])
    tank_to_collector_pipe = pipe.Pipe(temperature=INITIAL_SYSTEM_TEMPERATURE_VECTOR[3])

    # Instnatiate the hot-water pump.
    # htf_pump = process_pvt_system_data.pump_from_path(parsed_args.pump_data_file)

    # Intiailise the mains supply system.
    mains_supply = mains_power.MainsSupply.from_yaml(
        os.path.join(parsed_args.location, "utilities.yaml")
    )
    logger.info("Mains supply successfully instantiated: %s", mains_supply)

    # Set up the time iterator.
    num_months = (
        (parsed_args.initial_month if parsed_args.initial_month is not None else 1)
        - 1
        + parsed_args.months
    )  # [months]
    start_month = (
        parsed_args.initial_month
        if 1 <= parsed_args.initial_month <= 12
        else DEFAULT_INITIAL_DATE_AND_TIME.month
    )  # [months]
    initial_date_and_time = DEFAULT_INITIAL_DATE_AND_TIME.replace(
        hour=parsed_args.start_time, month=start_month
    )
    if parsed_args.days is None:
        final_date_and_time = initial_date_and_time + relativedelta(
            months=num_months % 12, years=num_months // 12
        )
    else:
        final_date_and_time = initial_date_and_time + relativedelta(
            days=parsed_args.days
        )

    # Set up a holder for information about the system.
    system_data: Dict[datetime.datetime, SystemData] = dict()

    # Set up a holder for the information about the final output of the system.
    # total_power_data = TotalPowerData()

    logger.debug(
        "Beginning itterative model:\n  Running from: %s\n  Running to: %s",
        str(initial_date_and_time),
        str(final_date_and_time),
    )

    logger.info(
        "System state before beginning run:\n%s\n%s\n%s\n%s",
        heat_exchanger,
        hot_water_tank,
        pvt_panel,
        weather_forecaster,
    )

    previous_run_temperature_vector: numpy.ndarray = numpy.asarray(  # type: ignore
        INITIAL_SYSTEM_TEMPERATURE_VECTOR
    )

    for run_number, date_and_time in enumerate(
        time_iterator(
            first_time=initial_date_and_time,
            last_time=final_date_and_time,
            resolution=parsed_args.resolution,
            timezone=pvt_panel.timezone,
        )
    ):

        logger.info(
            "Beginning internal run. Time: %s.\nPrevious temperature vector: T=\n%s",
            date_and_time.strftime("%d/%m/%Y %H:%M:%S"),
            previous_run_temperature_vector,
        )

        # Determine the current hot-water load.
        current_hot_water_load = (
            load_system[(load.ProfileType.HOT_WATER, date_and_time)]  # [litres/hour]
            / 3600  # [seconds/hour]
        )  # [kg/s]

        # Determine the current weather conditions.
        weather_conditions = weather_forecaster.get_weather(
            pvt_panel.latitude,
            pvt_panel.longitude,
            parsed_args.cloud_efficacy_factor,
            date_and_time,
        )

        coefficient_matrix = matrix.calculate_coefficient_matrix(
            1.2,
            current_hot_water_load,
            hot_water_tank,
            1.2,
            previous_run_temperature_vector,
            pvt_panel,
            parsed_args.resolution,
            weather_conditions,
        )

        resultant_vector = matrix.calculate_resultant_vector(
            current_hot_water_load,
            hot_water_tank,
            previous_run_temperature_vector,
            pvt_panel,
            parsed_args.resolution,
            weather_conditions,
        )

        logger.info(
            "Matrix equation computed.\nA =\n%s\nB =\n%s",
            str(coefficient_matrix),
            str(resultant_vector),
        )

        current_run_temperature_vector = linalg.solve(
            a=coefficient_matrix, b=resultant_vector
        )

        logger.info(
            "Temperatures successfully computed. Temperature vector: T =\n%s",
            current_run_temperature_vector,
        )

        system_data[run_number] = SystemData(
            date=date_and_time.strftime("%d/%m/%Y"),
            time=date_and_time.strftime("%H:%M:%S"),
            glass_temperature=current_run_temperature_vector[0, 0]
            - ZERO_CELCIUS_OFFSET,
            pv_temperature=current_run_temperature_vector[1, 0] - ZERO_CELCIUS_OFFSET,
            collector_temperature=current_run_temperature_vector[2, 0]
            - ZERO_CELCIUS_OFFSET,
            collector_input_temperature=current_run_temperature_vector[3, 0]
            - ZERO_CELCIUS_OFFSET,
            collector_output_temperature=current_run_temperature_vector[4, 0]
            - ZERO_CELCIUS_OFFSET,
            bulk_water_temperature=(
                current_run_temperature_vector[3, 0]
                + current_run_temperature_vector[4, 0]
            )
            / 2
            - ZERO_CELCIUS_OFFSET,
            ambient_temperature=weather_conditions.ambient_temperature
            - ZERO_CELCIUS_OFFSET,
            exchanger_temperature_drop=current_run_temperature_vector[3, 0]
            - current_run_temperature_vector[4, 0]
            if current_run_temperature_vector[4, 0]
            > current_run_temperature_vector[5, 0]
            else 0,
            tank_temperature=current_run_temperature_vector[5, 0] - ZERO_CELCIUS_OFFSET,
            sky_temperature=weather_conditions.sky_temperature - ZERO_CELCIUS_OFFSET,
        )

        previous_run_temperature_vector = current_run_temperature_vector

    # Save the output data from the run.
    _save_data(FileType.JSON, parsed_args.output, system_data)


if __name__ == "__main__":
    main(sys.argv[1:])
