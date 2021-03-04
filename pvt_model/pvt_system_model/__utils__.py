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

from typing import Any, Dict, Generator, Optional

from dataclasses import dataclass
from dateutil.relativedelta import relativedelta

import numpy
import yaml

from ..__utils__ import ProgrammerJudgementFault

__all__ = (
    "BaseDailyProfile",
    "CollectorParameters",
    "Date",
    "DivergentSolutionError",
    "InternalError",
    "InvalidDataError",
    "MissingDataError",
    "ResolutionMismatchError",
    "LayerParameters",
    "OpticalLayerParameters",
    "PVParameters",
    "PVT_SYSTEM_MODEL_LOGGER_NAME",
    "read_yaml",
    "time_iterator",
    "UtilityType",
    "WeatherConditions",
)


#############
# Constants #
#############

PVT_SYSTEM_MODEL_LOGGER_NAME = "pvt_model.pvt_system_model"

##############
# Exceptions #
##############


class DivergentSolutionError(Exception):
    """
    Raised when a divergent solution occurs.

    """

    def __init__(
        self,
        convergence_run_number: int,
        run_one_temperature_difference: float,
        run_one_temperature_vector: numpy.ndarray,
        run_two_temperature_difference: float,
        run_two_temperature_vector: numpy.ndarray,
    ) -> None:
        """
        Instantiate a :class:`DivergentSolutionError`.

        :param convergence_run_number:
            The number of runs attempted to reach a convergent solution.

        :param run_one_temperature_difference:
            The temperature difference between the i-2 and i-1 iterations.

        :param run_one_temperature_vector:
            The temperature vector computed at the i-1 iteration.

        :param run_two_temperature_difference:
            The temperature difference between the i-1 and i iterations.

        :param run_two_temperature_vector:
            The temperature vector computed at the i iteration.

        """

        super().__init__(
            "A divergent solution was found when attempting to compute the "
            "temperatures at the next time step:\n"
            f"Number of convergent runs attempted: {convergence_run_number}\n"
            f"Previous difference: {run_one_temperature_difference}\n"
            f"Current difference: {run_two_temperature_difference}\n"
            "Divergence is hence "
            f"{round(run_two_temperature_difference - run_one_temperature_difference, 2)}"
            " away from the current solution.\n"
            f"Previous solution temperatures:\n{run_one_temperature_vector}\n"
            f"Current solution temperatures:\n{run_two_temperature_vector}\n",
        )


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

    .. attribute:: ambient_tank_temperature
        The ambient temperature surrounding the hot-water tank, measured in Kelvin.

    .. attribute:: ambient_temperature
        The ambient temperature in

    .. attribute:: declination
        The angle of declination of the sun above the horizon

    .. attribute:: azimuthal_angle
        The azimuthal angle of the sun, defined clockwise from True North.

    .. attribute:: wind_speed
        The wind speed in meters per second.

    .. attribute:: mains_water_temperature
        The temperature of the mains water, measured in Kelvin.

    """

    # Private attributes:
    #
    # .. attribute:: _irradiance
    #   The solar irradiance in Watts per meter squared.
    #

    _irradiance: float
    ambient_tank_temperature: float
    ambient_temperature: float
    azimuthal_angle: float
    declination: float
    mains_water_temperature: float
    wind_speed: float

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

        NOTE: The equation from Ilaria's paper is used in stead of Maria's.

        :return:
            The convective heat transfer coefficient due to the wind, measured in Watts
            per meter squared Kelvin.

        """

        return 3.8 + 2 * self.wind_speed
        # return 4.5 + 2.9 * self.wind_speed

    def __repr__(self) -> str:
        """
        Return a nice representation of the weather conditions.

        :return:
            A nicely-formatted string containing weather conditions data.

        """

        return (
            "WeatherConditions("
            f"irradiance: {self.irradiance}, "
            f"declination: {self.declination}, "
            f"azimuthal_angle: {self.azimuthal_angle}, "
            f"wind_speed: {self.wind_speed}, "
            f"ambient_temperature: {self.ambient_temperature}, "
            f"sky_temperature: {self.sky_temperature}, "
            f"wind_heat_transfer_coefficient: {self.wind_heat_transfer_coefficient:2f}"
            ")"
        )


###################################
# PVT Layer Parameter Dataclasses #
###################################


@dataclass
class LayerParameters:
    """
    Contains parameters needed to instantiate a layer within the PV-T panel.

    .. attribute:: area
        The area of the layer, measured in meters squared.

    .. attribute:: conductivity
        The conductivity of the layer, measured in Watts per meter Kelvin.

    .. attribute:: density
        The density of the layer, measured in kilograms per meter cubed.

    .. attribute:: heat_capacity
        The heat capacity of the layer, measured in Joules per kilogram Kelvin.

    .. attribute:: mass
        The mass of the layer, measured in Kelvin.

    .. attribute:: thickness
        The thickness of the layer, measured in meters.

    """

    conductivity: float
    density: float
    heat_capacity: float
    thickness: float


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

    .. attribute:: htf_heat_capacity
        The heat capacity of the heat-transfer fluid through the collector, measured in
        Joules per kilogram Kelvin.

    .. attribute:: inner_pipe_diameter
        The diameter of the inner wall of the pipes, in meters.

    .. attribute:: length
        The legnth of the collector, measured in meters.

    .. attribute:: mass_flow_rate
        The mass flow rate of heat-transfer fluid through the collector. Measured in
        litres per hour.

    .. attribute:: number_of_pipes
        The number of pipes attached to the back of the thermal collector.
        NOTE: This parameter is very geography/design-specific, and will only be
        relevant/useful to the current design of collector being modeled. Namely, when
        multiple pipes flow linearly down the length of the collector, with the HTF
        taking a single pass through the collector.

    .. attribute:: outer_pipe_diameter
        The diameter of the outer wall of the pipes, in meters.

    """

    htf_heat_capacity: float
    inner_pipe_diameter: float
    length: float
    mass_flow_rate: float
    number_of_pipes: int
    outer_pipe_diameter: float


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


def read_yaml(yaml_file_path: str) -> Dict[Any, Any]:
    """
    Read in some yaml data and return it.

    :param yaml_file_path:
        The path to the yaml data to read in.

    :return:
        A `dict` containing the data read in from the yaml file.

    """

    logger = logging.getLogger(PVT_SYSTEM_MODEL_LOGGER_NAME)

    # Open the yaml data and read it.
    if not os.path.isfile(yaml_file_path):
        logger.error(
            "A YAML data file, '%s', could not be found. Exiting...", yaml_file_path
        )
        raise FileNotFoundError(yaml_file_path)
    with open(yaml_file_path) as f:
        try:
            data: Dict[Any, Any] = yaml.safe_load(f)
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
            hours=resolution // 3600,
            minutes=(resolution // 60) % 60,
            seconds=resolution % 60,
        )
