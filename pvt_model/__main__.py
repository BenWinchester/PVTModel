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
import math
import os
import sys

from argparse import Namespace
from typing import Dict, Optional, Tuple

import json
import numpy
import scipy
import yaml

from dateutil.relativedelta import relativedelta

from . import (
    argparser,
    efficiency,
    load,
    mains_power,
    process_pvt_system_data,
    pump,
    weather,
)
from .pvt_panel import pvt
from .__utils__ import (
    CarbonEmissions,
    DENSITY_OF_WATER,
    FileType,
    get_logger,
    HEAT_CAPACITY_OF_WATER,
    INITIAL_SYSTEM_TEMPERATURE,
    LOGGER_NAME,
    MissingParametersError,
    solar_heat_input,
    time_iterator,
    TotalPowerData,
    ZERO_CELCIUS_OFFSET,
)


# The temperature of hot-water required by the end-user, measured in Kelvin.
HOT_WATER_DEMAND_TEMP = 60 + ZERO_CELCIUS_OFFSET
# The initial date and time for the simultion to run from.
INITIAL_DATE_AND_TIME = datetime.datetime(2005, 1, 1, 0, 0)
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


def _temperature_vector_gradient(
    temperature_vector: Tuple[float, float, float, float, float],
    time: numpy.float64,
    parsed_args: Namespace,
    pvt_panel: pvt.PVT,
    weather_forecaster: weather.WeatherForecaster,
) -> Tuple[float, float, float, float, float]:
    """
    Computes the vector of the temperature gradients at some time t.

    :param temperature_vector:
        A `tuple` representing a vector containing:
        - the glass temperature,
        - the pv layer temperature,
        - the collector temperature,
        - the bulk-water temperature,
        - the hot-water tank tempreature.

        All of these temperature values are measured in Kelvin.

    :param time:
        The current time in seconds from the beginning of the model run.

    :param weather_forecaster:
        A :class:`weather.WeatherForecaster` containing weather information and exposing
        functions enabling the weather conditions to be calculated at some given time.

    :return:
        A `tuple` representing a vector containing:
        - the differential of the glass temperature with respect to time,
        - the differential of the pv layer temperature with respect to time,
        - the differential of the collector temperature with respect to time,
        - the differential of the bulk-water temperature with respect to time,
        - the differential of the hot-water tank temperature with respect to time.

    """

    # Unpack the temperature tuple.
    (
        glass_temperature,
        pv_temperature,
        collector_temperature,
        bulk_water_temperature,
        tank_temperature,
    ) = temperature_vector

    # Determine the environmental conditions at the current time step.
    weather_conditions = weather_forecaster.get_weather(
        pvt_panel.latitude,
        pvt_panel.longitude,
        parsed_args.cloud_efficacy_factor,
        _date_and_time_from_time_step(time),
    )

    bulk_water_temperature_gradient = (
        pvt_panel.convective_heat_transfer_coefficient  # [W/m^2*K]
        * pvt_panel.htf_surface_area  # [m^2]
        / (
            pvt_panel.htf_volume  # [m^3]
            * DENSITY_OF_WATER  # [kg/m^3]
            * pvt_panel.htf_heat_capacity  # [J/kg*K]
        )
    ) * (collector_temperature - bulk_water_temperature)

    pv_temperature_gradient = solar_heat_input(
        pvt_panel.pv_area,
        weather_conditions.solar_energy_input,
    )

    return (
        glass_temperature_gradient,
        pv_temperature_gradient,
        collector_temperature_gradient,
        bulk_water_temperature_gradient,
        tank_temperature_gradient,
    )


def main(args) -> None:
    """
    The main module for the code.

    :param args:
        The command-line arguments passed into the component.

    """

    logger.info("Beginning run of PVT model.\nCommand: %s", " ".join(args))

    # Parse the system arguments from the commandline.
    parsed_args = argparser.parse_args(args)

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
        INITIAL_SYSTEM_TEMPERATURE,
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
    htf_pump = process_pvt_system_data.pump_from_path(parsed_args.pump_data_file)

    # Intiailise the mains supply system.
    mains_supply = mains_power.MainsSupply.from_yaml(
        os.path.join(parsed_args.location, "utilities.yaml")
    )
    logger.info("Mains supply successfully instantiated: %s", mains_supply)

    # Loop through all times and iterate the system.
    input_water_temperature: float = (
        parsed_args.input_water_temperature + ZERO_CELCIUS_OFFSET
        if parsed_args.input_water_temperature is not None
        else INITIAL_SYSTEM_TEMPERATURE
    )  # [K]
    num_months = (
        (parsed_args.initial_month if parsed_args.initial_month is not None else 1)
        - 1
        + parsed_args.months
    )  # [months]
    start_month = (
        parsed_args.initial_month
        if 1 <= parsed_args.initial_month <= 12
        else INITIAL_DATE_AND_TIME.month
    )  # [months]
    first_date_and_time = INITIAL_DATE_AND_TIME.replace(
        hour=parsed_args.start_time, month=start_month
    )
    if parsed_args.days is None:
        final_date_and_time = first_date_and_time + relativedelta(
            months=num_months % 12, years=num_months // 12
        )
    else:
        final_date_and_time = first_date_and_time + relativedelta(days=parsed_args.days)

    # Set up a dictionary for storing the system data.
    system_data: Dict[int, SystemData] = dict.fromkeys(
        set(range((final_date_and_time - first_date_and_time).seconds))
    )

    # Set up a holder for the information about the final output of the system.
    total_power_data = TotalPowerData()

    logger.debug(
        "Beginning itterative model:\n  Running from: %s\n  Running to: %s",
        str(first_date_and_time),
        str(final_date_and_time),
    )

    logger.info(
        "System state before beginning run:\n%s\n%s\n%s\n%s",
        heat_exchanger,
        hot_water_tank,
        pvt_panel,
        weather_forecaster,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
