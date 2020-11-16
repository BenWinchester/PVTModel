#!/usr/bin/python3.7
########################################################################################
# weather.py - Computs daily weather characteristics.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The weather module for this PV-T model.

This module computes daily solar irradiance, based on the time of year, lattitude etc.,
and factors in the time of day, and cloud cover, to give an accurate estimate of the
solar irradiance at any given time.

Extensibly, it has the potential to compute the rainfail in mm/time_step s.t. the
cooling effect on panels can be estimated and included into the model as well.

"""

import time

from .__utils__ import WeatherConditions

__all__ = ("WeatherForecaster",)


class WeatherForecaster:
    """
    Represents a weather forecaster, determining weather conditions and irradiance.

    .. attribute:: irradiance
        The solar irradiance at any given time interval - returned as a property.

    """

    def __init__(self) -> None:
        """
        Instantiate a weather forecaster class.

        """

    def _cloud_cover(self, time_of_day: time.struct_time) -> int:
        """
        Computes the percentage cloud clover based on weather conditions.

        :param time_of_day:
            The time of day, used to calculate the irradiance.

        :return:
            An `int` giving the fraction of cloud cover.

        """

    def irradiance(self, time_of_day: time.struct_time) -> WeatherConditions:
        """
        Computes the solar irradiance based on weather conditions at the time of day.

        :param time_of_day:
            The time of day, used to calculate the irradiance.

        :return:
            A :class:`__utils__.WeatherConditions` giving the solar irradiance, in
            watts, and the angles, both azimuthal and declination, of the sun's position
            in the sky.

        """

        # * Based on the time, compute the sun's position in the sky, making sure to
        # * account for the seasonal variation.
        irradiance: float = 0

        # * Factor in the weather conditions and cloud cover to compute the current
        # * solar irradiance.
        declination: float = 0
        azimuthal_angle: float = 0

        # * Compute the wind speed and ambient temperature
        wind_speed: float = 0
        ambient_temperature: float = 0

        # * Return all of these in a WeatherConditions variable.

        return WeatherConditions(
            irradiance, declination, azimuthal_angle, wind_speed, ambient_temperature
        )
