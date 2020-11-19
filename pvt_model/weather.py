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

import calendar
import datetime
import random

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import pysolar
import yaml

from .__utils__ import MissingParametersError, WeatherConditions, read_yaml

__all__ = ("WeatherForecaster",)


# The resolution to which random numbers are generated
RAND_RESOLUTION = 100


@dataclass
class _MonthlyWeatherData:
    """
    Contains weather data for a month.

    .. attribute:: num_days
        The average number of days in the month.

    .. attribute:: cloud_cover
        The probabilty that any given day within the month is cloudy.

    .. attribute:: rainy_days
        The average number of days in the month for which rain occurs.

    .. attribute:: day_temp
        The average daytime temperature, measured in Kelvin, for the month.

    .. attribute:: night_temp
        The average nighttime temperature, measured in Kelvin, for the month.

    """

    month_name: str
    num_days: float
    cloud_cover: float
    rainy_days: float
    day_temp: float
    night_temp: float

    @classmethod
    def from_yaml(cls, month_name: str, monthly_weather_data: Dict[Any, Any]) -> Any:
        """
        Checks for present fields and instantiates from YAML data.

        :param month_name:
            A `str` giving a three-letter representation of the month.

        :param monthly_weather_data:
            A `dict` containing the weather data for the month, extracted raw from the
            weather YAML file.

        :return:
            An instance of the class.

        """

        try:
            return cls(
                month_name,
                monthly_weather_data["num_days"],
                monthly_weather_data["cloud_cover"],
                monthly_weather_data["rainy_days"],
                monthly_weather_data["day_temp"],
                monthly_weather_data["night_temp"],
            )
        except KeyError as e:
            raise MissingParametersError(
                "WeatherForecaster",
                "Missing fields in YAML file. Error: {}".format(str(e)),
            ) from None


class WeatherForecaster:
    """
    Represents a weather forecaster, determining weather conditions and irradiance.

    """

    # Private attributes:
    #
    # .. attribute:: _month_abbr_to_num
    #   A mapping from month abbreviated name (eg, "jan", "feb" etc) to the number of
    #   the month in the year.
    #
    # .. attribute:: _monthly_weather_data
    #   A `dict` mapping month number to :class:`_MonthlyWeatherData` instances
    #   containing weather information for that month.
    #
    # .. attribute:: _solar_insolation
    #   The solar insolation, measured in Watts per meter squared, that would hit the
    #   UK on a clear day with no other factors present.
    #

    _month_abbr_to_num = {
        name.lower(): num
        for num, name in enumerate(calendar.month_abbr)
        if num is not None
    }

    def __init__(
        self,
        solar_insolation: float,
        monthly_weather_data: Dict[str, Union[str, float]],
    ) -> None:
        """
        Instantiate a weather forecaster class.

        :param solar_insolation:
            The solar insolation, measured in Watts per meter squared, that would hit
            the UK on a clear day with no other factors present.

        :param monthly_weather_data:
            The monthly weather data, extracted raw from the weather data YAML file.

        """

        self._solar_insolation = solar_insolation

        self._monthly_weather_data = {
            self._month_abbr_to_num[month]: _MonthlyWeatherData.from_yaml(
                _MonthlyWeatherData, month_data
            )
            for month, month_data in monthly_weather_data.values()
        }

    @classmethod
    def from_yaml(cls, weather_data_path: str) -> Any:
        """
        Instantiate a :class:`WeatherForecaster` from a path to a weather-data file.

        :param weather_data_path:
            The path to the weather-data file.

        :return:
            A :class:`WeatherForecaster` instance.

        """

        # Call out to the __utils__ module to read the yaml data.
        data = read_yaml(weather_data_path)

        # * Check that all months are specified.
        try:
            solar_insolation = data.pop("solar_insolation")
        except KeyError:
            raise MissingParametersError(
                "WeatherForecaster",
                "The solar insolation param is missing from {}.".format(
                    weather_data_path
                ),
            ) from None

        # Instantiate and return a Weather Forecaster based off of this weather data.

        return cls(solar_insolation, data)

    def _cloud_cover(self, date_and_time: datetime.datetime) -> float:
        """
        Computes the cloud clover based on the time of day and various factors.

        :param date_and_time:
            The date and time of day, used to determine which month the cloud cover
            should be retrieved for.

        :return:
            The fractional effect that the cloud cover has to reduce

        """

        # Extract the cloud cover probability for the month.
        cloud_cover_prob = self._monthly_weather_data[date_and_time.month]

        # Generate a random number between 1 and 0 for that month based on this factor.
        rand_prob: float = random.randrange(0, RAND_RESOLUTION, 1) / RAND_RESOLUTION

        # Determine what effect the cloudy (or not cloudy) conditions have on the solar
        # insolation. Generate a fractional reduction based on this.
        # Return this number
        return cloud_cover_prob * rand_prob

    def _get_solar_angles(
        self, latitude: float, longitude: float, date_and_time: datetime.datetime
    ) -> Tuple[float, float]:
        """
        Determine the azimuthal_angle (right-angle) and declination of the sun.

        :param latitude:
            The latitude of the PV-T set-up.

        :param longitude:
            The longitude of the PV-T set-up.

        :param date_and_time:
            The current date and time.

        :return:
            A `tuple` containing the azimuthal angle and declination of the sun at the
            given date and time.

        """

        return (
            pysolar.solar.get_azimuth(latitude, longitude, date_and_time),
            pysolar.solar.get_altitude(latitude, longitude, date_and_time),
        )

    def irradiance(
        self, latitude: float, longitude: float, date_and_time: datetime.datetime
    ) -> WeatherConditions:
        """
        Computes the solar irradiance based on weather conditions at the time of day.

        :param latitude:
            The latitude of the PV-T set-up.

        :param longitude:
            The longitude of the PV-T set-up.

        :param date_and_time:
            The date and time of day, used to calculate the irradiance.

        :return:
            A :class:`__utils__.WeatherConditions` giving the solar irradiance, in
            watts per meter squared, and the angles, both azimuthal and declination, of
            the sun's position in the sky.

        """

        # Based on the time, compute the sun's position in the sky, making sure to
        # account for the seasonal variation.
        declination, azimuthal_angle = self._get_solar_angles(
            latitude, longitude, date_and_time
        )

        # Factor in the weather conditions and cloud cover to compute the current solar
        # irradiance.
        irradiance: float = self._solar_insolation * self._cloud_cover(date_and_time)

        # * Compute the wind speed and ambient temperature
        wind_speed: float = 0
        ambient_temperature: float = 0

        # Return all of these in a WeatherConditions variable.
        return WeatherConditions(
            irradiance, declination, azimuthal_angle, wind_speed, ambient_temperature
        )
