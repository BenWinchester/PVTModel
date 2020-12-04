#!/usr/bin/python3.7
########################################################################################
# __main__.py - The main module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The main module for the PV-T model.

This module coordinates the time-stepping of the itterative module, calling out to the
various components where necessary, as well as reading in the various data files and
processing the command-line arguments that define the scope of the model.

"""

import argparse
import dataclasses
import datetime
import logging  # pylint: disable=unused-import
import os
import sys

from typing import Any, Dict, Generator, Optional

import json

from dateutil.relativedelta import relativedelta

from . import efficiency, exchanger, load, pvt, tank, weather
from .__utils__ import (
    InvalidDataError,
    MissingDataError,
    MissingParametersError,
    BackLayerParameters,
    CollectorParameters,
    OpticalLayerParameters,
    PVParameters,
    read_yaml,
    HEAT_CAPACITY_OF_WATER,
    LOGGER_NAME,
    ZERO_CELCIUS_OFFSET,
)


# Name of the weather data file.
WEATHER_DATA_FILENAME = "weather.yaml"
# Name of the load data file.
LOAD_DATA_FILENAME = "loads_watts.yaml"
# The initial date and time for the simultion to run from.
INITIAL_DATE_AND_TIME = datetime.datetime(2020, 1, 1, 0, 0)
# The initial temperature for the system to be instantiated at, measured in Kelvin.
INITIAL_SYSTEM_TEMPERATURE = 293  # [K]
# The temperature of hot-water required by the end-user, measured in Kelvin.
HOT_WATER_DEMAND_TEMP = 60 + ZERO_CELCIUS_OFFSET


# * Arg-parsing method


def time_iterator(
    *,
    first_time: datetime.datetime,
    last_time: datetime.datetime,
    resolution: int,
    timezone: datetime.timezone,
) -> Generator[datetime.datetime, None, None]:
    """
    A generator function for looping through various times.

    :param first_time:
        The first time to be returned from the function.

    :param last_time:
        The last time, which, when reached, should cause the generator to stop.

    :param resolution:
        The time step, in minutes, to run the iterator for.

    :param timezone:
        The timezone of the PV-T set-up.

    :return:
        A :class:`datetime.datetime` corresponding to the date and time at each point
        being itterated through.

    """

    current_time = first_time
    while current_time < last_time:
        yield current_time.replace(tzinfo=timezone)
        current_time += relativedelta(minutes=resolution)


@dataclasses.dataclass
class SystemData:
    """
    Contains PVT system data at a given time step.

    .. attribute:: date
        The date.

    .. attribute:: time
        The time.

    .. attribute:: ambient_temperature
        The ambient temperature, in Celcius.

    .. attribute:: sky_temperature
        The sky temperature, radiatively, in Celcius.

    .. attribute:: solar_irradiance
        The solar irradiance in Watts per meter squared.

    .. attribute:: normal_irradiance
        The solar irradiance, in Watts, normal to the panel.

    .. attribute:: glass_temperature
        The temperatuer of the glass layer of the panel, measured in Celcius.

    .. attribute:: pv_temperature
        The temperature of the PV layer of the panel, measured in Celcius. This is set
        to `None` if no PV layer is present.

    .. attribute:: pv_efficiency
        The efficiency of the PV panel, defined between 0 and 1. This is set to `None`
        if no PV layer is present.

    .. attribute:: collector_temperature
        The temperature of the thermal collector, measured in Celcius.

    .. attribute:: collector_output_temperature
        The temperature of the HTF outputted from the collector, measured in Celcius.

    .. attribute:: collector_temperature_gain
        The temperature gain of the HTF through the collector, measured in Celcius.

    .. attribute:: tank_temperature
        The temperature of the water within the hot-water tank, measured in Celcius.

    .. attribute:: tank_output_temperature
        The temperature of the water outputted from the hot-water tank, measured in
        Celcius.

    .. attribute:: tank_heat_addition
        The heat added to the tank, in Watts.

    .. attribute:: electrical_load
        The load (demand) placed on the PV-T panel's electrical output, measured in
        Watts.

    .. attribute:: thermal_load
        The load (demand) placed on the hot-water tank's thermal output, measured in
        Watts.

    .. attribute:: thermal_output
        The thermal output from the PV-T system supplied - this is really a combnination
        of the demand required and the temperature of the system, measured in Watts.

    .. attribute:: auxiliary_heating
        The additional energy needed to be supplied to the system through the auxiliary
        heater when the tank temperature is below the required thermal output
        temperature, measured in Watts.

    .. attribute:: gross_electrical_output
        The electrical power produced by the panel, measured in Watts.

    .. attribute:: net_electrical_output
        The electrical power produced by the panel which is in excess of the demand
        required by the household. IE: gross - demand. This is measured in Watts.

    .. attribute:: dc_electrical
        The electrical demand covered, defined between 0 and 1.

    .. attribute:: dc_thermal
        The thermal demand covered, defined between 0 and 1.

    """

    date: str
    time: str
    ambient_temperature: float
    sky_temperature: float
    solar_irradiance: float
    normal_irradiance: float
    glass_temperature: Optional[float]
    pv_temperature: Optional[float]
    pv_efficiency: Optional[float]
    collector_temperature: float
    collector_output_temperature: float
    collector_temperature_gain: float
    tank_temperature: float
    tank_output_temperature: float
    tank_heat_addition: float
    electrical_load: float
    thermal_load: float
    thermal_output: float
    auxiliary_heating: float
    gross_electrical_output: float
    net_electrical_output: float
    dc_electrical: Optional[float] = None
    dc_thermal: Optional[float] = None

    def __str__(self) -> str:
        """
        Output a pretty picture.

        :return:
            A `str` containing the system data info.

        """

        return (
            f"System Data at {self.date}::{self.time}: "
            f"T_g/C {round(self.glass_temperature, 2)}"
            f"T_pv/C {round(self.pv_temperature, 2)} T_c/C "
            f"{round(self.collector_temperature, 2)} T_t/C "
            f"{round(self.tank_temperature, 2)} pv_eff {round(self.pv_efficiency, 2)} "
            f"aux {round(self.auxiliary_heating, 2)} "
            f"dc_e {round(self.dc_electrical, 2)} dc_t {round(self.dc_thermal, 2)}"
        )


def _get_logger(logger_name: str) -> logging.Logger:
    """
    Set-up and return a logger.

    :param logger_name:
        The name of the logger to instantiate.

    :return:
        The logger for the component.

    """

    # Create a logger with the current component name.
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Create a file handler which logs even debug messages.
    if os.path.exists("pvt_simulator.log"):
        os.rename("pvt_simulator.log", "pvt_simulator.log.1")
    fh = logging.FileHandler("pvt_simulator.log")
    fh.setLevel(logging.DEBUG)
    # Create a console handler with a higher log level.
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # Create a formatter and add it to the handlers.
    formatter = logging.Formatter(
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def _parse_args(args) -> argparse.Namespace:
    """
    Parse command-line arguments.

    :param args:
        The command-line arguments to parse.

    :return:
        The parsed arguments in an :class:`argparse.Namespace`.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cloud-efficacy-factor",
        "-c",
        type=float,
        help="The effect that the cloud cover has, rated between 0 (no effect) and 1.",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        help="The number of days to run the simulation for. Overrides 'months'.",
    )
    parser.add_argument(
        "--exchanger-data-file",
        "-e",
        help="The location of the Exchanger system YAML data file.",
    )
    parser.add_argument(
        "--initial-month",
        "-i",
        type=int,
        help="The first month for which the simulation will be run, expressed as an "
        "int. The default is 1, corresponding to January.",
    )
    parser.add_argument(
        "--input-water-temperature",
        "-it",
        help="The input water temperature to instantiate the system, measured in "
        "Celcius. Defaults to 20 Celcius if not provided.",
    )
    parser.add_argument(
        "--location", "-l", help="The location for which to run the simulation."
    )
    parser.add_argument(
        "--months",
        "-m",
        help="The number of months for which to run the simulation. Default is 12.",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--no-pv",
        action="store_true",
        default=False,
        help="Used to specify a PV-T panel with no PV layer: ie, a Thermal collector.",
    )
    parser.add_argument(
        "--number-of-people",
        "-n",
        type=int,
        help="The number of household members to consider in this model.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="The output file to save data to. This should be of JSON format.",
    )
    parser.add_argument(
        "--pvt-data-file", "-p", help="The location of the PV-T system YAML data file."
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        help="The time-step resolution, in minutes, for which to run the simulation.",
    )
    parser.add_argument(
        "--start-time",
        "-st",
        type=int,
        default=0,
        help="The start time, in hours, at which to begin the simulation during the day",
    )
    parser.add_argument(
        "--tank-data-file",
        "-t",
        help="The location of the Hot-Water Tank system YAML data file.",
    )

    return parser.parse_args(args)


def glass_params_from_data(
    area: float, glass_data: Dict[str, Any]
) -> OpticalLayerParameters:
    """
    Generate a :class:`OpticalLayerParameters` containing glass-layer info from data.

    :param area:
        The area of the PV-T system, measured in meters squared.

    :param glass_data:
        The raw glass data extracted from the YAML data file.

    :return:
        The glass data, as a :class:`__utils__.OpticalLayerParameters`, ready to
        instantiate a :class:`pvt.Glass` layer instance.

    """

    try:
        return OpticalLayerParameters(
            glass_data["mass"]  # [kg]
            if "mass" in glass_data
            else glass_data["density"]  # [kg/m^3]
            * glass_data["thickness"]  # [m]
            * area,  # [m^2]
            glass_data["heat_capacity"],  # [J/kg*K]
            area,  # [m^2]
            glass_data["thickness"],  # [m]
            INITIAL_SYSTEM_TEMPERATURE,  # [K]
            glass_data["transmissivity"],  # [unitless]
            glass_data["absorptivity"],  # [unitless]
            glass_data["emissivity"],  # [unitless]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed glass-layer data provided. Potential problem: Glass mass "
            "must be specified, either as 'mass' or 'density' and 'area' and "
            f"'thickness' params: {str(e)}"
        ) from None


def pv_params_from_data(area: float, pv_data: Dict[str, Any]) -> PVParameters:
    """
    Generate a :class:`PVParameters` containing PV-layer info from data.

    :param area:
        The area of the PV-T system, measured in meters squared.

    :param pv_data:
        The raw PV data extracted from the YAML data file.

    :return:
        The PV data, as a :class:`__utils__.PVParameters`, ready to instantiate a
        :class:`pvt.PV` layer instance.

    """

    try:
        return PVParameters(
            pv_data["mass"]  # [kg]
            if "mass" in pv_data
            else pv_data["density"]  # [kg/m^3]
            * area  # [m^2]
            * pv_data["thickness"],  # [m]
            pv_data["heat_capacity"],  # [J/kg*K]
            area,  # [m^2]
            pv_data["thickness"],  # [m]
            INITIAL_SYSTEM_TEMPERATURE,  # [K]
            pv_data["transmissivity"],  # [unitless]
            pv_data["absorptivity"],  # [unitless]
            pv_data["emissivity"],  # [unitless]
            pv_data["reference_efficiency"],  # [unitless]
            pv_data["reference_temperature"],  # [K]
            pv_data["thermal_coefficient"],  # [K^-1]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed PV-layer data provided. Potential problem: PV mass must be"
            "specified, either as 'mass' or 'density' and 'area' and 'thickness' "
            f"params: {str(e)}"
        ) from None


def collector_params_from_data(
    area: float,
    initial_collector_htf_tempertaure: float,
    collector_data: Dict[str, Any],
) -> CollectorParameters:
    """
    Generate a :class:`CollectorParameters` containing collector-layer info from data.

    The HTF is assumed to be water unless the HTF heat capacity is supplied in the
    collector data.

    :param area:
        The area of the PV-T system, measured in meters squared.

    :param initial_collector_htf_tempertaure:
        The initial temperature of heat-transfer fluid in the collector.

    :param collector_data:
        The raw collector data extracted from the YAML data file.

    :return:
        The collector data, as a :class:`__utils__.CollectorParameters`, ready to
        instantiate a :class:`pvt.Collector` layer instance.

    """

    try:
        return CollectorParameters(
            collector_data["mass"]  # [kg]
            if "mass" in collector_data
            else area  # [m^2]
            * collector_data["density"]  # [kg/m^3]
            * collector_data["thickness"],  # [m]
            collector_data["heat_capacity"],  # [J/kg*K]
            area,  # [m^2]
            collector_data["thickness"],  # [m]
            INITIAL_SYSTEM_TEMPERATURE,  # [K]
            initial_collector_htf_tempertaure,  # [K]
            collector_data["mass_flow_rate"],  # [Litres/hour]
            collector_data["htf_heat_capacity"]  # [J/kg*K]
            if "htf_heat_capacity" in collector_data
            else HEAT_CAPACITY_OF_WATER,  # [J/kg*K]
            collector_data["pump_power"],  # [W]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed collector-layer data provided. Potential problem: collector"
            "mass must be specified, either as 'mass' or 'density' and 'area' and "
            f"'thickness' params: {str(e)}"
        ) from None


def back_params_from_data(
    area: float, back_data: Dict[str, Any]
) -> BackLayerParameters:
    """
    Generate a :class:`BackLayerParameters` containing back-layer info from data.

    :param area:
        The area of the PV-T system, measured in meters squared.

    :param back_data:
        The raw back data extracted from the YAML data file.

    :return:
        The back data, as a :class:`__utils__.BackLayerParameters`, ready to
        instantiate a :class:`pvt.BackPlater` layer instance.

    """

    try:
        return BackLayerParameters(
            back_data["mass"]  # [kg]
            if "mass" in back_data
            else back_data["density"]  # [kg/m^3]
            * area  # [m^2]
            * back_data["thickness"],  # [m]
            back_data["heat_capacity"],  # [J/kg*K]
            area,  # [m^2]
            back_data["thickness"],  # [m]
            INITIAL_SYSTEM_TEMPERATURE,  # [K]
            back_data["thermal_conductivity"],  # [W/m*K]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed back-layer data provided. Potential problem: back-layer"
            "mass must be specified, either as 'mass' or 'density' and 'area' and "
            f"'thickness' params. Missing param: {str(e)}"
        ) from None


def pvt_panel_from_path(
    pvt_data_file: str,
    initial_collector_htf_tempertaure: float,
    pv_layer_included: bool,
) -> pvt.PVT:
    """
    Generate a :class:`pvt.PVT` instance based on the path to the data file.

    :param pvt_data_file:
        The path to the pvt data file.

    :param initial_collector_htf_tempertaure:
        The intial temperature, measured in Kelvin, of the HTF within the thermal
        collector.

    :param pv_layer_included:
        Whether or not a PV layer is included in the panel.

    :return:
        A :class:`pvt.PVT` instance representing the PVT panel.

    """

    # Set up the PVT module
    pvt_data = read_yaml(pvt_data_file)

    glass_parameters = glass_params_from_data(
        pvt_data["pvt_system"]["area"], pvt_data["glass"]
    )
    pv_parameters = (
        pv_params_from_data(pvt_data["pvt_system"]["area"], pvt_data["pv"])
        if "pv" in pvt_data
        else None
    )
    collector_parameters = collector_params_from_data(
        pvt_data["pvt_system"]["area"],  # [m^2]
        initial_collector_htf_tempertaure,  # [K]
        pvt_data["collector"],
    )
    back_parameters = back_params_from_data(
        pvt_data["pvt_system"]["area"], pvt_data["back"]
    )

    try:
        pvt_panel = pvt.PVT(
            pvt_data["pvt_system"]["latitude"],  # [deg]
            pvt_data["pvt_system"]["longitude"],  # [deg]
            glass_parameters,
            collector_parameters,
            back_parameters,
            pvt_data["pvt_system"]["air_gap_thickness"],  # [m]
            pvt_data["pvt_system"]["pv_to_collector_conductance"],  # [W/m^2*K]
            datetime.timezone(
                datetime.timedelta(hours=int(pvt_data["pvt_system"]["timezone"]))
            ),
            pv_layer_included="pv" in pvt_data and pv_layer_included,
            pv_parameters=pv_parameters if pv_layer_included else None,
            tilt=pvt_data["pvt_system"]["tilt"]  # [deg]
            if "tilt" in pvt_data["pvt_system"]
            else None,
            azimuthal_orientation=pvt_data["pvt_system"][
                "azimuthal_orientation"
            ]  # [deg]
            if "azimuthal_orientation" in pvt_data["pvt_system"]
            else None,
            horizontal_tracking=pvt_data["pvt_system"]["horizontal_tracking"],
            vertical_tracking=pvt_data["pvt_system"]["vertical_tracking"],
        )
    except KeyError as e:
        raise MissingParametersError(
            "PVT", f"Missing parameters when instantiating the PV-T system: {str(e)}"
        ) from None
    except TypeError as e:
        raise InvalidDataError(
            "PVT Data File", f"Error parsing data types - type mismatch: {str(e)}"
        ) from None

    return pvt_panel


def heat_exchanger_from_path(exchanger_data_file: str) -> exchanger.Exchanger:
    """
    Generate a :class:`exchanger.Exchanger` instance based on the path to the data file.

    :param exchanger_data_file:
        The path to the exchanger data file.

    :return:
        A :class:`exchanger.Exchanger` instance representing the heat exchanger.

    """

    exchanger_data = read_yaml(exchanger_data_file)
    try:
        return exchanger.Exchanger(float(exchanger_data["efficiency"]))  # [unitless]
    except KeyError as e:
        raise MissingDataError(
            f"The file '{exchanger_data_file}' "
            f"is missing exchanger parameters: efficiency: {str(e)}"
        ) from None
    except ValueError as e:
        raise InvalidDataError(
            exchanger_data_file,
            "The exchanger efficiency must be a floating point integer.",
        ) from None


def hot_water_tank_from_path(tank_data_file: str, mains_water_temp: float) -> tank.Tank:
    """
    Generate a :class:`tank.Tank` instance based on the path to the data file.

    :param tank_data_file:
        The path to the tank data file.

    :param mains_water_temp:
        The mains-water temperature, measured in Kelvin.

    :return:
        A :class:`tank.Tank` instance representing the hot-water tank.

    """

    tank_data = read_yaml(tank_data_file)
    try:
        return tank.Tank(
            mains_water_temp,  # [K]
            float(tank_data["mass"]),  # [kg]
            HEAT_CAPACITY_OF_WATER,  # [J/kg*K]
            float(tank_data["area"]),  # [m^2]
            float(tank_data["heat_loss_coefficient"]),  # [W/m^2*K]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all data needed to instantiate the tank class was provided. "
            f"File: {tank_data_file}. Error: {str(e)}"
        ) from None
    except ValueError as e:
        raise InvalidDataError(
            tank_data_file,
            "Tank data variables provided must be floating point integers.",
        ) from None


def _save_data(system_data: Dict[int, SystemData], output_file_name: str) -> None:
    """
    Save data when called.

    :param system_data:
        The data to save.

    :param output_file_name:
        The destination file name.

    """

    with open(output_file_name, "w+") as f:
        json.dump(
            {key: dataclasses.asdict(value) for key, value in system_data.items()},
            f,
            indent=4,
        )


def main(args) -> None:  # pylint: disable=too-many-locals
    """
    The main module for the code.

    :param args:
        The command-line arguments passed into the component.

    """

    # Set up logging with a file handler etc.
    logger = _get_logger(LOGGER_NAME)
    logger.info("Logger successfully instantiated.")

    # Parse the system arguments from the commandline.
    parsed_args = _parse_args(args)

    # Check that the output file is specified, and that it doesn't already exist.
    if parsed_args.output is None or parsed_args.output == "":
        logger.error(
            "An output filename must be provided on the command-line interface."
        )
        raise MissingParametersError(
            "Command-Line Interface", "An output file name must be provided."
        )
    if not parsed_args.output.endswith(".json"):
        logger.error("The output filename must end with .json.")
        raise Exception("The output file must be in json format.")
    if os.path.isfile(parsed_args.output):
        logger.debug("The output file specified already exists. Moving...")
        os.rename(parsed_args.output, f"{parsed_args.output}.1")

    # Set-up the weather and load modules with the weather and load probabilities.
    weather_forecaster = weather.WeatherForecaster.from_yaml(
        os.path.join(parsed_args.location, WEATHER_DATA_FILENAME)
    )
    load_system = load.LoadSystem.from_yaml(
        os.path.join(parsed_args.location, LOAD_DATA_FILENAME),
        parsed_args.number_of_people,
    )
    logger.info(
        "Weather forecaster and load system successfully instantiated:\n  %s\n  %s",
        str(weather_forecaster),
        str(load_system),
    )

    # Initialise the PV-T class, tank, exchanger, etc..
    pvt_panel = pvt_panel_from_path(
        parsed_args.pvt_data_file, INITIAL_SYSTEM_TEMPERATURE, not parsed_args.no_pv
    )
    logger.info("PV-T panel successfully instantiated: %s", str(pvt_panel))
    heat_exchanger = heat_exchanger_from_path(parsed_args.exchanger_data_file)
    hot_water_tank = hot_water_tank_from_path(
        parsed_args.tank_data_file, weather_forecaster.mains_water_temp
    )

    # Set up a way to store system data.
    system_data: Dict[datetime.datetime, SystemData] = dict()

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

    logger.debug(
        "Beginning itterative model:\n  Running from: %s\n  Running to: %s",
        str(first_date_and_time),
        str(final_date_and_time),
    )
    for run_number, date_and_time in enumerate(
        time_iterator(
            first_time=first_date_and_time,
            last_time=final_date_and_time,
            resolution=parsed_args.resolution,
            timezone=pvt_panel.timezone,
        )
    ):

        # Generate weather and load conditions from the load and weather classes.
        current_weather = weather_forecaster.get_weather(
            *pvt_panel.coordinates, parsed_args.cloud_efficacy_factor, date_and_time
        )

        # NOTE: The electrical load will have the same units as the entries in the load
        # data file passed in on the command-line interface. If the file contains data
        # entries in units kilo Watts, then the electrical load here will have units of
        # kilo Watts.
        current_electrical_load = load_system.get_load_from_time(
            parsed_args.resolution, load.ProfileType.ELECTRICITY, date_and_time
        )  # [Watts]
        current_hot_water_load = load_system.get_load_from_time(
            parsed_args.resolution, load.ProfileType.HOT_WATER, date_and_time
        )  # [litres/time step]

        # logger.info(
        #     "Weather and load conditions determined at time %s:\n  %s"
        #     "\n  Electrical load: %s\n  Thermal load: %s",
        #     date_and_time,
        #     current_weather,
        #     current_electrical_load,
        #     current_hot_water_load,
        # )

        # Call the pvt module to generate the new temperatures at this time step.
        output_water_temperature = pvt_panel.update(
            input_water_temperature, parsed_args.resolution, current_weather
        )  # [K]

        # Propogate this information through to the heat exchanger and pass in the
        # tank s.t. it updates the tank correctly as well.
        # The tank heat gain here is measured in Joules.
        input_water_temperature, tank_heat_gain = heat_exchanger.update(  # [K], [J]
            hot_water_tank,
            output_water_temperature,  # [K]
            pvt_panel.mass_flow_rate * parsed_args.resolution * 60,  # [kg]
            pvt_panel.htf_heat_capacity,  # [J/kg*K]
        )

        # Compute the new tank temperature after supplying this demand
        tank_output_water_temp = hot_water_tank.update(  # [K]
            tank_heat_gain,  # [J]
            parsed_args.resolution,  # [minutes]
            current_hot_water_load,  # [litres/time step]
            weather_forecaster.mains_water_temp,  # [K]
            current_weather.ambient_temperature,  # [K]
        )

        # Determine various efficiency factors
        # The auxiliary heating will be measured in Watts.
        auxiliary_heating = (
            1  # [kg/litres]
            * current_hot_water_load  # [litres/time step]
            * HEAT_CAPACITY_OF_WATER  # [J/kg*K]
            * (HOT_WATER_DEMAND_TEMP - tank_output_water_temp)  # [K]
        ) / (
            parsed_args.resolution * 60  # We need to convert from Joules to Watts
        )  # [time step in seconds]

        dc_electrical = (
            efficiency.dc_electrical(
                electrical_output=pvt_panel.electrical_output(
                    parsed_args.resolution, current_weather
                ),
                electrical_losses=pvt_panel.pump_power,
                electrical_demand=current_electrical_load,
            )
            * 100
        )

        dc_thermal = (
            efficiency.dc_thermal(
                thermal_output=current_hot_water_load
                * HEAT_CAPACITY_OF_WATER
                * (tank_output_water_temp - weather_forecaster.mains_water_temp),
                thermal_demand=current_hot_water_load
                * HEAT_CAPACITY_OF_WATER
                * (HOT_WATER_DEMAND_TEMP - weather_forecaster.mains_water_temp),
            )
            * 100
        )

        # Store the information in the dictionary mapping between time step and data.
        system_data[run_number] = SystemData(
            date=f"{date_and_time.day}/{date_and_time.month}/{date_and_time.year}",
            time=f"{date_and_time.hour}:{date_and_time.minute}",
            ambient_temperature=current_weather.ambient_temperature
            - ZERO_CELCIUS_OFFSET,
            sky_temperature=current_weather.sky_temperature - ZERO_CELCIUS_OFFSET,
            solar_irradiance=current_weather.irradiance,
            normal_irradiance=pvt_panel.get_solar_irradiance(current_weather),
            glass_temperature=pvt_panel.glass_temperature - ZERO_CELCIUS_OFFSET,
            pv_temperature=pvt_panel.pv_temperature - ZERO_CELCIUS_OFFSET,
            pv_efficiency=pvt_panel.electrical_efficiency,
            collector_temperature=pvt_panel.collector_temperature - ZERO_CELCIUS_OFFSET,
            collector_output_temperature=pvt_panel.collector_output_temperature
            - ZERO_CELCIUS_OFFSET,
            collector_temperature_gain=pvt_panel.collector_output_temperature
            - input_water_temperature,
            tank_temperature=hot_water_tank.temperature - ZERO_CELCIUS_OFFSET,
            tank_output_temperature=tank_output_water_temp - ZERO_CELCIUS_OFFSET,
            tank_heat_addition=tank_heat_gain / (parsed_args.resolution * 60),
            electrical_load=current_electrical_load,
            thermal_load=current_hot_water_load
            * HEAT_CAPACITY_OF_WATER
            * 50
            / (parsed_args.resolution * 60),
            thermal_output=current_hot_water_load
            * HEAT_CAPACITY_OF_WATER
            * (tank_output_water_temp - 10 - ZERO_CELCIUS_OFFSET)
            / (parsed_args.resolution * 60),
            auxiliary_heating=auxiliary_heating,
            gross_electrical_output=pvt_panel.electrical_output(
                parsed_args.resolution, current_weather
            ),
            net_electrical_output=pvt_panel.electrical_output(
                parsed_args.resolution, current_weather
            )
            - current_electrical_load,
            dc_electrical=dc_electrical,
            dc_thermal=dc_thermal,
        )

        logger.info(
            "System data determined at time %s:\n%s",
            date_and_time,
            system_data[run_number],
        )

    # Dump the data generated to the output JSON file.
    _save_data(system_data, parsed_args.output)

    # * Potentially generate some plots, and at least save the data.


if __name__ == "__main__":
    main(sys.argv[1:])
