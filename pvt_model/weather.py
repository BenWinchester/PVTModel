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

import datetime

from typing import Any, Dict

import yaml

from .__utils__ import WeatherConditions

__all__ = ("WeatherForecaster",)


class WeatherForecaster:
    """
    Represents a weather forecaster, determining weather conditions and irradiance.

    """

    # Private attributes:
    #
    # .. attribute:: _monthly_temperature_averages
    #   A `dict` mapping month number to the temperature average for that month,
    #   measured in Kelvin.
    #
    # .. attribute:: _monthly_cloud_cover_averages
    #   A `dict` mapping month number to the average cloud cover for that month,
    #   with the cloud cover being a measured of the cloud cover in some, as-of-yet to-
    #   be-determined units.
    #

    def __init__(
        self,
        monthly_temperature_averages: Dict[int, float],
        monthly_cloud_cover_averages: Dict[int, float],
    ) -> None:
        """
        Instantiate a weather forecaster class.

        """

        self._monthly_temperature_averages = monthly_temperature_averages
        self._monthly_cloud_cover_averages = monthly_cloud_cover_averages

    @classmethod
    def from_yaml(cls, weather_data_path: str) -> Any:
        """
        Instantiate a :class:`WeatherForecaster` from a path to a weather-data file.

        :param weather_data_path:
            The path to the weather-data file.

        :return:
            A :class:`WeatherForecaster` instance.

        """

        # * Call out to the __utils__ module to read the yaml data.

        # * Check that all fields needed are specified.

        # * Instantiate and return a Weather Forecaster based off of this weather data.

        return cls(dict(), dict())

    def _cloud_cover(self, date_and_time: datetime.datetime) -> int:
        """
        Computes the percentage cloud clover based on weather conditions.

        :param date_and_time:
            The date and time of day, used to determine which month the cloud cover
            should be retrieved for.

        :return:
            An `int` giving the fraction of cloud cover.

        """

    def irradiance(self, date_and_time: datetime.datetime) -> WeatherConditions:
        """
        Computes the solar irradiance based on weather conditions at the time of day.

        :param date_and_time:
            The date and time of day, used to calculate the irradiance.

        :return:
            A :class:`__utils__.WeatherConditions` giving the solar irradiance, in
            watts per meter squared, and the angles, both azimuthal and declination, of
            the sun's position in the sky.

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
