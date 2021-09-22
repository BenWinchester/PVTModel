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

from typing import Any, Dict, Generator, List, Union

from dataclasses import dataclass
from dateutil.relativedelta import relativedelta

import numpy

from .constants import SPECIFIC_GAS_CONSTANT_OF_AIR


from ..__utils__ import (
    ProgrammerJudgementFault,
)

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
    "TEMPERATURE_FOLDERNAME",
    "time_iterator",
    "UtilityType",
    "WeatherConditions",
    "WEATHER_DATA_FILENAME",
)


#############
# Constants #
#############

# Name of the logger used
PVT_SYSTEM_MODEL_LOGGER_NAME = "pvt_system.{tag}_run_{run_number}"
# Folder containing the solar irradiance profiles
SOLAR_IRRADIANCE_FOLDERNAME = "solar_irradiance_profiles"
# Folder containing the temperature profiles
TEMPERATURE_FOLDERNAME = "temperature_profiles"
# Name of the weather data file.
WEATHER_DATA_FILENAME = "weather.yaml"

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
        run_one_temperature_vector: Union[List[float], numpy.ndarray],
        run_two_temperature_difference: float,
        run_two_temperature_vector: Union[List[float], numpy.ndarray],
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

        run_one_temperature_vector_string = ""
        for entry in run_one_temperature_vector:
            run_one_temperature_vector_string += ", {}".format(str(entry))

        run_two_temperature_vector_string = ""
        for entry in run_two_temperature_vector:
            run_two_temperature_vector_string += ", {}".format(str(entry))

        super().__init__(
            "A divergent solution was found when attempting to compute the "
            "temperatures at the next time step:\n"
            f"Number of convergent runs attempted: {convergence_run_number}\n"
            f"Previous difference: {run_one_temperature_difference}\n"
            f"Current difference: {run_two_temperature_difference}\n"
            "Divergence is hence "
            f"{round(run_two_temperature_difference - run_one_temperature_difference, 2)}"
            " away from the current solution.\n"
            f"Previous solution temperatures:\n{run_one_temperature_vector_string}\n"
            f"Current solution temperatures:\n{run_two_temperature_vector_string}\n",
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

    .. attribute:: mains_water_temperature
        The temperature of the mains water, measured in Kelvin.

    .. attribute:: pressure
        The air pressure, measured in Pascals = Newtons per meter squared.

    .. attribute:: wind_speed
        The wind speed in meters per second.


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
    pressure: float
    wind_speed: float

    @property
    def density_of_air(self) -> float:
        """
        The density of air varies as a function of temperature.

        The data for the density is obtained from:
        https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470516430.app2

        :return:
            The density of air, measured in kilograms per meter cubed.

        """

        return self.pressure / (SPECIFIC_GAS_CONSTANT_OF_AIR * self.ambient_temperature)

    @property
    def dynamic_viscosity_of_air(self) -> float:
        """
        The dynamic viscosity of air varies as a function of temperature.

        The data for the dynamic viscosity is obtained from:
        https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470516430.app2

        :return:
            The dynamic viscosity of air, measured in kilograms per meter second.

        """

        return (1.458 * (10 ** (-6)) * (self.ambient_temperature ** 1.5)) / (
            self.ambient_temperature + 110.4
        )

    @property
    def heat_capacity_of_air(self) -> float:
        """
        Return the heat capacity of air in Joules perkilogram Kelvin.

        The heat capacity of air varies with a function of temperature and is given by
        an empirically-derived formula.

        https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470516430.app2

        :return:
            The heat capacity of air in Joules per kilogram Kelvin.

        """

        return 1002.5 + 275 * (10 ** (-6)) * (self.ambient_temperature - 200) ** 2

    @property
    def irradiance(self) -> float:
        """
        The irradiance should only be definied if the sun is above the horizon.

        :return:
            The solar irradiance, adjusted for the day-night cycle, measured in Watts
            per meter squared.

        """

        if self.declination > 0:
            return self._irradiance
        return 0

    @property
    def kinematic_viscosity_of_air(self) -> float:
        """
        The kinematic viscosity of air varies as a function of temperature.

        The data for the dynamic viscosity is obtained from:
        https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470516430.app2

        :return:
            The kinematic viscosity of air, measured in meters squared per second.

        """

        return self.dynamic_viscosity_of_air / self.density_of_air

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
    def thermal_conductivity_of_air(self) -> float:
        """
        The thermal conductivity of air varies as a function of temperature.

        The data for the thermal conductivity is obtained from:
        https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470516430.app2

        :return:
            The thermal conductivity of air, measured in Watts per meter Kelvin.

        """

        # This more accurate equation is not used by the paper.
        # return (0.02646 * self.ambient_temperature ** 1.5) / (
        #     self.ambient_temperature + 254.4 * (10 ** (-12 / self.ambient_temperature))
        # )

        # The reference suggests this equation is accurate to 1%.
        return 0.02646 * (self.ambient_temperature / 300) ** 0.8646

    @property
    def thermal_expansivity_of_air(self) -> float:
        """
        The thermal expansion coefficient of air varies as a function of temperature.

        The data for the thermal expansion coefficient is obtained from:
        https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470516430.app2

        :return:
            The thermal expansion coefficient of air, measured in Kelvin to the minus 1.

        """

        return 1 / self.ambient_temperature

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
            f"ambient_temperature: {self.ambient_temperature:.3f}K, "
            f"azimuthal_angle: {self.azimuthal_angle}deg, "
            f"declination: {self.declination}deg, "
            f"density: {self.density_of_air:.3f}kg/m^3, "
            f"dynamic_viscosity: {self.dynamic_viscosity_of_air:.3f}kg/m*s, "
            f"heat_capacity: {self.heat_capacity_of_air}:.3fJ/kg*K, "
            f"irradiance: {self.irradiance:.3f}W/m^2, "
            f"kinematic_viscosity: {self.kinematic_viscosity_of_air:.3f}m^2/s, "
            f"sky_temperature: {self.sky_temperature:.3f}K, "
            f"thermal_conductivity: {self.thermal_conductivity_of_air:.3f}W/m*K, "
            f"thermal_expansion_coefficient: {self.thermal_expansivity_of_air:.3f}K^-1, "
            f"wind_heat_transfer_coefficient: {self.wind_heat_transfer_coefficient:2f}W/m*K, "
            f"wind_speed: {self.wind_speed:.3f}m/s, "
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
    Contains parameters needed to instantiate a absorber layer within the PV-T panel.

    .. attribute:: htf_heat_capacity
        The heat capacity of the heat-transfer fluid through the absorber, measured in
        Joules per kilogram Kelvin.

    .. attribute:: inner_pipe_diameter
        The diameter of the inner wall of the pipes, in meters.

    .. attribute:: length
        The legnth of the absorber, measured in meters.

    .. attribute:: mass_flow_rate
        The mass flow rate of heat-transfer fluid through the absorber. Measured in
        litres per hour.

    .. attribute:: number_of_pipes
        The number of pipes attached to the back of the thermal absorber.
        NOTE: This parameter is very geography/design-specific, and will only be
        relevant/useful to the current design of absorber being modeled. Namely, when
        multiple pipes flow linearly down the length of the absorber, with the HTF
        taking a single pass through the absorber.

    .. attribute:: outer_pipe_diameter
        The diameter of the outer wall of the pipes, in meters.

    .. attribute:: pipe_density
        The density of the material making up the pipes, measured in kilograms per meter
        cubed.

    """

    htf_heat_capacity: float
    inner_pipe_diameter: float
    length: float
    mass_flow_rate: float
    number_of_pipes: int
    outer_pipe_diameter: float
    pipe_density: float


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
