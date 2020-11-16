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

from typing import NamedTuple

__all__ = ("WeatherConditions",)


class WeatherConditions(NamedTuple):
    """
    Contains information about the various weather conditions at any given time.

    .. attribute:: irradiance
        The solar irradiance in Watts.

    .. attribute:: declination
        The angle of declination of the sun above the horizon

    .. attribute:: azimuthal_angle
        The azimuthal angle of the sun, defined clockwise from True North.

    .. attribute:: wind_speed
        The wind speed in meters per second.

    .. attribute:: ambient_temperature
        The ambient temperature in

    """

    irradiance: float
    declination: float
    azimuthal_angle: float
    wind_speed: float
    ambient_temperature: float

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
