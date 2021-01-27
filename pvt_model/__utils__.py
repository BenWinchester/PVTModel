#!/usr/bin/python3.7
########################################################################################
# __utils__.py - The utility module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The utility module for the PV-T model.

This module contains common functionality, strucutres, and types, to be used by the
various modules throughout the PVT model.

"""

import datetime
import enum
import logging
import os

from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

from dateutil.relativedelta import relativedelta

import yaml

__all__ = (
    "BackLayerParameters",
    "BaseDailyProfile",
    "CarbonEmissions",
    "CollectorParameters",
    "Date",
    "FileType",
    "FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR",
    "HEAT_CAPACITY_OF_WATER",
    "get_logger",
    "GraphDetail",
    "INITIAL_SYSTEM_TEMPERATURE",
    "INITIAL_TANK_TEMPERATURE",
    "InternalError",
    "InvalidDataError",
    "LOGGER_NAME",
    "MissingDataError",
    "MissingParametersError",
    "NUSSELT_NUMBER",
    "ResolutionMismatchError",
    "LayerParameters",
    "OpticalLayerParameters",
    "ProgrammerJudgementFault",
    "PVParameters",
    "read_yaml",
    "STEFAN_BOLTZMAN_CONSTANT",
    "THERMAL_CONDUCTIVITY_OF_AIR",
    "THERMAL_CONDUCTIVITY_OF_WATER",
    "time_iterator",
    "TotalPowerData",
    "UtilityType",
    "WeatherConditions",
    "ZERO_CELCIUS_OFFSET",
)


#############
# Constants #
#############

# The temperature of absolute zero in Kelvin, used for converting Celcius to Kelvin and
# vice-a-versa.
ZERO_CELCIUS_OFFSET: float = 273.15

# The free convective, heat-transfer coefficient of air. This varies, and potentially
# could be extended to the weather module and calculated on the fly depending on various
# environmental conditions etc.. This is measured in Watts per meter squared
# Kelvin.
FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR: int = 25
# The heat capacity of water, measured in Joules per kilogram Kelvin.
HEAT_CAPACITY_OF_WATER: int = 4182
# The initial temperature for the system to be instantiated at, measured in Kelvin.
INITIAL_SYSTEM_TEMPERATURE = 283  # [K]
# The initial temperature of the hot-water tank, at which it should be instantiated,
# measured in Kelvin.
INITIAL_TANK_TEMPERATURE = ZERO_CELCIUS_OFFSET + 34.75  # [K]
# The name used for the internal logger.
LOGGER_NAME = "my_first_pvt_model"
# The Nusselt number of the flow is given as 6 in Maria's paper.
NUSSELT_NUMBER: float = 6
# The Stefan-Boltzman constant, given in Watts per meter squared Kelvin to the four.
STEFAN_BOLTZMAN_CONSTANT: float = 5.670374419 * (10 ** (-8))
# The convective, heat-transfer coefficienct of water. This varies (a LOT), and is
# measured in units of Watts per meter squared Kelvin.
# This is determined by the following formula:
#   Nu = h_w * D / k_w
# where D is the diameter of the pipe, in meters, and k_w is the thermal conductivity of
# water.
# FIXME - I think a constant should be inserted here.

# The thermal conductivity of air is measured in Watts per meter Kelvin.
# ! This is defined at 273 Kelvin.
THERMAL_CONDUCTIVITY_OF_AIR: float = 0.024
# The thermal conductivity of water is obtained from
# http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/thrcn.html
THERMAL_CONDUCTIVITY_OF_WATER: float = 0.6  # [W/m*K]


##############
# Exceptions #
##############


class InternalError(Exception):
    """
    Used for internal error handling and catching where no information is needed.

    """


class InvalidDataError(Exception):
    """
    Raised when some proviced data is of the wrong format.

    """

    def __init__(self, data_file_name: str, message: str) -> None:
        """
        Instantiate an Invalid Data Error.

        :param data_file_name:
            The name of the data file containing the invalid data entry.

        :param message:
            An error message to be appended.

        """

        super().__init__(
            "Invalid data error. "
            "The file '{}' contained data of the wrong format: {}".format(
                data_file_name, message
            )
        )


class MissingDataError(Exception):
    """
    Raised when data is requested from a class that is missing.

    """

    def __init__(self, message: str) -> None:
        """
        Instantiate a Missing Data Error.

        :param message:
            The message to append for the user.

        """

        super().__init__(
            "Data requested from a class could not be found: {}".format(message)
        )


class MissingParametersError(Exception):
    """
    Raised when not all parameters have been specified that are needed to instantiate.

    """

    def __init__(self, class_name: str, message: str) -> None:
        """
        Instantiate a missing parameters error.

        :param class_name:
            The class for which insufficient parameters were specified.

        :param message:
            An appended message to display to the user.

        """

        super().__init__(
            f"Missing parameters when initialising a '{class_name}' class: {message}."
        )


class ProgrammerJudgementFault(Exception):
    """
    Raised when an error is hit due to poor programming.

    """

    def __init__(self, message: str) -> None:
        """
        Instantiate a programmer judgement fault error.

        :param message:
            A message to append when displaying the error to the user.

        """

        super().__init__(f"A programmer judgement fault has occurred: {message}")


class ResolutionMismatchError(Exception):
    """
    Raised when an error occurs attempting to match up data and simulation resolutions.

    """

    def __init__(self, message: str) -> None:
        """
        Instantiate a resolution mismatch error.

        :param message:
            An appended message to display to the user.

        """

        super().__init__(
            "Due to the nature of the load data, an integer multiple of resolutions, "
            "or divsions of resolutions, must be supplied with the '--resolution' or "
            "'-r' flag.\nI appreciate that this is poor coding, but at least I took "
            "the time to write a custom exception for it :p .\n Error message: "
            f"{message}"
        )


##############################
# Functions and Data Classes #
##############################


class BaseDailyProfile:
    """
    Represents a day's profile.

    """

    # Private Attributes:
    # .. attribute:: _profile
    #   A mapping of the time of day to a floating point value.
    #

    def __init__(self, profile: Dict[datetime.time, float] = None) -> None:
        """
        Instantiate the daily profile class..

        :param profile:
            The daily profile as a mapping between the time of day and the floating
            point value.

        """

        if profile is None:
            profile = dict()

        if not isinstance(profile, dict):
            raise ProgrammerJudgementFault(
                "The input daily profile provided is not a mapping of the correct type."
            )

        self._profile = profile

    def __getitem__(self, index: datetime.time) -> float:
        """
        Return an irradiance value from the profile.

        :param index:
            The index of the item to return, must be a valid :class:`datetime.time`
            instance.

        :return:
            The profile's value at that time.

        """

        # If the index is in the profile, return the index.
        if index in self._profile:
            return self._profile[index]

        # If the index is not in the profile, then the closest value needs to be
        # determined. If there is a tie, this does not matter.
        delta_t_to_t_map = {
            (
                abs(
                    time.hour * 3600
                    + time.minute * 60
                    + time.second
                    - (index.hour * 3600 + index.minute * 60 + index.second)
                )
            ): time
            for time in self._profile
        }
        return self._profile[delta_t_to_t_map[min(delta_t_to_t_map)]]

    def __setitem__(self, index: datetime.time, value: float) -> None:
        """
        Sets an item in the profile.

        :param index:
            The index, i.e., :class:`datetime.time` instance, for which to set the
            value.

        :param value:
            The value to set.

        """

        self._profile[index] = value

    def update(self, profile: Dict[datetime.time, float]) -> None:
        """
        Updates the internal profile with the mapping provided.

        :param profile:
            The profile to add to the currently-stored internal profile.

        """

        if self._profile is None:
            self._profile = profile
        else:
            self._profile.update(profile)


@dataclass
class CarbonEmissions:
    """
    Contains information about the carbon emissions produced.

    .. attribute:: electrical_carbon_produced
        The amount of CO2 equivalent produced, measured in kilograms.

    .. attribute:: electrical_carbon_saved
        The amount of CO2 saved by using PV-T, measured in kilograms.

    .. attribute:: heating_carbon_produced
        The amount of CO2 equivalent produced, measured in kilograms.

    .. attribute:: heating_carbon_saved
        The amount of CO2 equivalent saved by using the PV-T, measured in kilograms.

    """

    electrical_carbon_produced: float
    electrical_carbon_saved: float
    heating_carbon_produced: float
    heating_carbon_saved: float


@dataclass
class Date:
    """
    Represents a date, containing informaiton about the month and day.

    .. attribute:: day
        The day of the month, ranging from 1 to 31, expressed as an `int`.

    .. attribute:: month
        The month of the year, expressed as an `int`.

    """

    day: int
    month: int

    def __hash__(self) -> int:
        """
        Returns a hash of the :class:`Date`.

        This is calulated using the inbuilt hashability of the datetime.date module.

        :return:
            A unique hash.

        """

        return hash(datetime.date(2000, self.month, self.day))

    @classmethod
    def from_date(cls, date: datetime.date) -> Any:
        """
        Instantiates a :class:`Date` instance based on a :class:`datetime.date` object.

        :param date:
            The date from which to instantiate this :class:`Date` instance.

        :return:
            An instantiated :class:`Date` instanced based on the information attached
            to the :class:`datetime.date` instance passed in.

        """

        return cls(date.day, date.month)


class FileType(enum.Enum):
    """
    Tells what type of file is being used for the data.

    .. attribute:: YAML
        A YAML file is being used.

    .. attribute:: JSON
        A JSON file is being used.

    """

    YAML = 0
    JSON = 1


class GraphDetail(enum.Enum):
    """
    The level of detail to go into when graphing.

    .. attribute:: highest
        The highest level of detail - all data points are plotted.

    .. attribute:: high
        A "high" level of detail, to be determined by the analysis script.

    .. attribute:; medium
        A "medium" level of detail, to be determined by the analysis script.

    .. attribute:: low
        A "low" level of detail, to be determined by the analysis script.

    .. attribute:: lowest
        The lowest level of detail, with points only every half an hour.

    """

    highest = 0
    high = 2880
    medium = 720
    low = 144
    lowest = 48


@dataclass
class TotalPowerData:
    """
    Contains information about the total power generated by the system.

    .. attribute:: electricity_supplied
        The electricity supplied, measured in Joules.

    .. attribute:: electricity_demand
        The electricity demand, measured in Joules.

    .. attribute:: heating_supplied
        The heatimg supplied, measured in Joules.

    .. attribute:: heating_demand
        The heating demand, measured in Joules.

    """

    electricity_supplied: float = 0
    electricity_demand: float = 0
    heating_supplied: float = 0
    heating_demand: float = 0

    def increment(
        self,
        electricity_supplied_incriment,
        electricity_demand_incriment,
        heating_supplied_increment,
        heating_demand_increment,
    ) -> None:
        """
        Updates the values by incrementing with those supplied.

        :param electricity_supplied_increment:
            Electricity supplied to add, measured in Joules.
        :param electricity_demand_increment:
            Electricity demand to add, measured in Joules.
        :param heating_supplied_increment:
            Heating supplied to add, measured in Joules.
        :param heating_demand_increment:
            Heating demand to add, measured in Joules.

        """

        self.electricity_supplied += electricity_supplied_incriment
        self.electricity_demand += electricity_demand_incriment
        self.heating_supplied += heating_supplied_increment
        self.heating_demand += heating_demand_increment


class UtilityType(enum.Enum):
    """
    Contains information about the type of mains utility being used.

    """

    gas = 0
    electricity = 1


@dataclass
class WeatherConditions:
    """
    Contains information about the various weather conditions at any given time.

    .. attribute:: irradiance
        The solar irradiance in Watts per meter squared.

    .. attribute:: declination
        The angle of declination of the sun above the horizon

    .. attribute:: azimuthal_angle
        The azimuthal angle of the sun, defined clockwise from True North.

    .. attribute:: wind_speed
        The wind speed in meters per second.

    .. attribute:: ambient_temperature
        The ambient temperature in

    """

    _irradiance: float
    declination: float
    azimuthal_angle: float
    wind_speed: float
    ambient_temperature: float

    @property
    def irradiance(self) -> float:
        """
        The irradiance should only be definied if the sun is above the horizon.

        :return:
            The solar irradiance, adjusted for the day-night cycle.

        """

        if self.declination > 0:
            return self._irradiance
        return 0

    @property
    def sky_temperature(self) -> float:
        """
        Determines the radiative temperature of the sky.

        The "sky," as a black body, has a radiative temperature different to that of the
        surrounding air, or the ambient temperature. This function converts between them
        and outputs the sky's radiative temperature.

        :return:
            The radiative temperature of the "sky" in Kelvin.

        """

        return 0.0552 * (self.ambient_temperature ** 1.5)

    @property
    def wind_heat_transfer_coefficient(self) -> float:
        """
        Determines the convective heat transfer coefficient, either free, or forced.

        In the absence of any wind, the "free" wind_heat_transfer_coefficient is
        returned. If there is wind present, then this parameter is known as the "forced"
        wind_heat_transfer_coefficient.

        :return:
            The convective heat transfer coefficient due to the wind, measured in Watts
            per meter squared Kelvin.

        """

        return 4.5 + 2.9 * self.wind_speed

    def __repr__(self) -> str:
        """
        Return a nice representation of the weather conditions.

        :return:
            A nicely-formatted string containing weather conditions data.

        """

        return (
            "WeatherConditions(irradiance: {}, declination: {}, ".format(
                self.irradiance, self.declination
            )
            + "azimuthal_angle: {}, wind_speed: {}, ambient_temperature: {}, ".format(
                self.azimuthal_angle, self.wind_speed, self.ambient_temperature
            )
            + "sky_temperature: {}, wind_heat_transfer_coefficient: {0:2f})".format(
                self.sky_temperature, self.wind_heat_transfer_coefficient
            )
        )


###################################
# PVT Layer Parameter Dataclasses #
###################################


@dataclass
class LayerParameters:
    """
    Contains parameters needed to instantiate a layer within the PV-T panel.

    .. attribute:: mass
        The mass of the layer, measured in Kelvin.

    .. attribute:: heat_capacity
        The heat capacity of the layer, measured in Joules per kilogram Kelvin.

    .. attribute:: area
        The area of the layer, measured in meters squared.

    .. attribute:: thickness
        The thickness of the layer, measured in meters.

    .. attribute:: temperature
        The temperature at which to initialise the layer, measured in Kelvin.

    """

    mass: float
    heat_capacity: float
    area: float
    thickness: float
    temperature: float


@dataclass
class BackLayerParameters(LayerParameters):
    """
    Contains parameters needed to instantiate the back layer of the PV-T panel.

    .. attribute:: conductance
        The conductance of layer (to the environment/its surroundings), measured in
        Watts per meter squared Kelvin.

    """

    conductivity: float


@dataclass
class OpticalLayerParameters(LayerParameters):
    """
    Contains parameters needed to instantiate a layer with optical properties.

    .. attribute:: transmissivity
        The transmissivity of the layer: a dimensionless number between 0 (nothing is
        transmitted through the layer) and 1 (all light is transmitted).

    .. attribute:: absorptivity
        The absorptivity of the layer: a dimensionless number between 0 (nothing is
        absorbed by the layer) and 1 (all light is absorbed).

    .. attribute:: emissivity
        The emissivity of the layer; a dimensionless number between 0 (nothing is
        emitted by the layer) and 1 (the layer re-emits all incident light).

    """

    transmissivity: float
    absorptivity: float
    emissivity: float


@dataclass
class CollectorParameters(OpticalLayerParameters):
    """
    Contains parameters needed to instantiate a collector layer within the PV-T panel.

    .. attribute:: length
        The legnth of the collector, measured in meters.

    .. attribute:: number_of_pipes
        The number of pipes attached to the back of the thermal collector.
        NOTE: This parameter is very geography/design-specific, and will only be
        relevant/useful to the current design of collector being modeled. Namely, when
        multiple pipes flow linearly down the length of the collector, with the HTF
        taking a single pass through the collector.

    .. attribute:: output_water_temperature
        The temperature, in Kelvin, of water outputted by the layer.

    .. attribute:: pipe_diameter
        The diameter of the pipe, in meters.

    .. attribute:: mass_flow_rate
        The mass flow rate of heat-transfer fluid through the collector. Measured in
        litres per hour.

    .. attribute:: htf_heat_capacity
        The heat capacity of the heat-transfer fluid through the collector, measured in
        Joules per kilogram Kelvin.

    .. attribute:: pump_power
        The electrical power, in Watts, consumed by the water pump in the collector.

    """

    length: float
    number_of_pipes: float
    output_water_temperature: float
    pipe_diameter: float
    mass_flow_rate: float
    htf_heat_capacity: float
    pump_power: float


@dataclass
class PVParameters(OpticalLayerParameters):
    """
    Contains parameters needed to instantiate a PV layer within the PV-T panel.

    .. attribute:: reference_efficiency
        The efficiency of the PV layer at the reference temperature. Thie value varies
        between 1 (corresponding to 100% efficiency), and 0 (corresponding to 0%
        efficiency)

    .. attribute:: reference_temperature
        The referencee temperature, in Kelvin, at which the reference efficiency is
        defined.

    .. attribute:: thermal_coefficient
        The thermal coefficient for the efficiency of the panel.

    """

    reference_efficiency: float
    reference_temperature: float
    thermal_coefficient: float


####################
# Helper functions #
####################


def get_logger(logger_name: str) -> logging.Logger:
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
    if os.path.exists(f"{logger_name}.log"):
        os.rename(f"{logger_name}.log", f"{logger_name}.log.1")
    fh = logging.FileHandler(f"{logger_name}.log")
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


def read_yaml(yaml_file_path: str) -> Dict[Any, Any]:
    """
    Read in some yaml data and return it.

    :param yaml_file_path:
        The path to the yaml data to read in.

    :return:
        A `dict` containing the data read in from the yaml file.

    """

    logger = logging.getLogger(LOGGER_NAME)

    # Open the yaml data and read it.
    if not os.path.isfile(yaml_file_path):
        logger.error(
            "A YAML data file, '%s', could not be found. Exiting...", yaml_file_path
        )
        raise FileNotFoundError(yaml_file_path)
    with open(yaml_file_path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.parser.ParserError as e:
            logger.error("Failed to read YAML file '%s'.", yaml_file_path)
            print(f"Failed to parse YAML. Internal error: {str(e)}")
            raise

    logger.info("Data successfully read from '%s'.", yaml_file_path)
    return data


def time_iterator(
    *,
    first_time: datetime.datetime,
    last_time: datetime.datetime,
    internal_resolution: int,
    timezone: datetime.timezone,
) -> Generator[datetime.datetime, None, None]:
    """
    A generator function for looping through various times.

    :param first_time:
        The first time to be returned from the function.

    :param last_time:
        The last time, which, when reached, should cause the generator to stop.

    :param internal_resolution:
        The time step, in seconds, for which the simulation should be run before saving.

    :param timezone:
        The timezone of the PV-T set-up.

    :return:
        A :class:`datetime.datetime` corresponding to the date and time at each point
        being itterated through.

    """

    current_time = first_time
    while current_time < last_time:
        yield current_time.replace(tzinfo=timezone)
        current_time += relativedelta(
            hours=internal_resolution // 3600,
            minutes=internal_resolution // 60,
            seconds=internal_resolution % 60,
        )
