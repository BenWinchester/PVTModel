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
import logging
import math
import random

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import pysolar

from .__utils__ import (
    ZERO_CELCIUS_OFFSET,
    LOGGER_NAME,
    MissingParametersError,
    WeatherConditions,
    read_yaml,
)

__all__ = ("WeatherForecaster",)


# A parameter used in modelling weather data curves.
G_CONST = 1

# The resolution to which random numbers are generated
RAND_RESOLUTION = 100

logger = logging.getLogger(LOGGER_NAME)


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

    .. attribute:: sunrise
        The sunrise time. Can be set later.

    .. attribute:: sunset
        The sunset time. Can be set later.

    """

    month_name: str
    num_days: float
    cloud_cover: float
    rainy_days: float
    day_temp: float
    night_temp: float
    sunrise: Optional[datetime.time] = None
    sunset: Optional[datetime.time] = None

    def __repr__(self) -> str:
        """
        Return a standard representation of the class.

        :return:
            A nicely-formatted monthly-weather data string.

        """

        return "_MonthlyWeatherData(month: {}, num_days: {}, cloud_cover: {}, ".format(
            self.month_name,
            self.num_days,
            self.cloud_cover,
        ) + "rainy_days: {}, day_temp: {}, night_temp: {})".format(
            self.rainy_days,
            self.day_temp,
            self.night_temp,
        )

    @classmethod
    def from_yaml(
        cls, month_name: str, monthly_weather_data: Dict[str, Union[str, float]]
    ) -> Any:
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
                float(monthly_weather_data["num_days"]),
                float(monthly_weather_data["cloud_cover"]),
                float(monthly_weather_data["rainy_days"]),
                float(monthly_weather_data["day_temp"]) + ZERO_CELCIUS_OFFSET,
                float(monthly_weather_data["night_temp"]) + ZERO_CELCIUS_OFFSET,
            )
        except KeyError as e:
            raise MissingParametersError(
                "WeatherForecaster",
                "Missing fields in YAML file. Error: {}".format(str(e)),
            ) from None


def _get_solar_angles(
    latitude: float, longitude: float, date_and_time: datetime.datetime
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


def _get_sunrise(
    latitude: float,
    longitude: float,
    date_and_time: datetime.datetime,
) -> datetime.time:
    """
    Determine the sunrise time for the month.

    :param latitude:
        The latitude of the PV-T set-up.

    :param longitude:
        The longitude of the PV-T set-up.

    :param date_and_time:
        The current date and time.

    :return:
        The time of sunrise for the month, returned as a :class:`datetime.time`.

    """

    _, declination = _get_solar_angles(latitude, longitude, date_and_time)
    if declination > 0:
        return date_and_time.time()
    return _get_sunrise(
        latitude,
        longitude,
        date_and_time.replace(hour=date_and_time.hour + 1),
    )


def _get_sunset(
    latitude: float,
    longitude: float,
    date_and_time: datetime.datetime,
    declination=0,
) -> datetime.time:
    """
    Determine the sunset time for the month.

    :param latitude:
        The latitude of the PV-T set-up.

    :param longitude:
        The longitude of the PV-T set-up.

    :param date_and_time:
        The current date and time.

    :return:
        The time of sunset for the month, returned as a :class:`datetime.time`.

    """

    _, declination = _get_solar_angles(latitude, longitude, date_and_time)
    if declination > 0:
        return date_and_time.time()
    return _get_sunset(
        latitude,
        longitude,
        date_and_time.replace(hour=date_and_time.hour - 1),
    )


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
        mains_water_temp: float,
        monthly_weather_data: Dict[str, Dict[str, Union[str, float]]],
    ) -> None:
        """
        Instantiate a weather forecaster class.

        :param solar_insolation:
            The solar insolation, measured in Watts per meter squared, that would hit
            the location on a clear day with no other factors present.

        :param mains_water_temp:
            The mains water temperature, measured in Kelvin.

        :param monthly_weather_data:
            The monthly weather data, extracted raw from the weather data YAML file.

        """

        self._solar_insolation = solar_insolation

        self.mains_water_temp = mains_water_temp + ZERO_CELCIUS_OFFSET

        self._monthly_weather_data = {
            self._month_abbr_to_num[month]: _MonthlyWeatherData.from_yaml(
                month, month_data  # type: ignore
            )
            for month, month_data in monthly_weather_data.items()
        }

    def __repr__(self) -> str:
        """
        Return a nice-looking representation of the weather forecaster.

        :return:
            A nicely-formatted string giving information about the weather forecaster.

        """

        return "WeatherForecaster(solar_insolation: {}, mains_water_temp: {}, ".format(
            self._solar_insolation, self.mains_water_temp
        ) + "num_months: {})".format(len(self._monthly_weather_data.keys()))

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
            logger.error(
                "Weather forecaster from %s is missing 'solar_insolation' data.",
                weather_data_path,
            )
            raise MissingParametersError(
                "WeatherForecaster",
                "The solar insolation param is missing from {}.".format(
                    weather_data_path
                ),
            ) from None

        try:
            mains_water_temp = data.pop("mains_water_temp")
        except KeyError:
            logger.error(
                "Weather forecaster from %s is missing 'mains_water_temp' data.",
                weather_data_path,
            )
            raise MissingParametersError(
                "WeatherForecaster",
                "The mains water temperature param is missing from {}.".format(
                    weather_data_path
                ),
            ) from None

        # Instantiate and return a Weather Forecaster based off of this weather data.

        return cls(solar_insolation, mains_water_temp, data)

    def _cloud_cover(
        self, cloud_efficacy_factor: float, date_and_time: datetime.datetime
    ) -> float:
        """
        Computes the cloud clover based on the time of day and various factors.

        :param cloud_efficacy_factor:
            The extend to which cloud cover affects the sunlight. This is multiplied by
            a random number which further reduces the effect of the cloud cover in
            reducing the sunlight.

        :param date_and_time:
            The date and time of day, used to determine which month the cloud cover
            should be retrieved for.

        :return:
            The fractional effect that the cloud cover has to reduce

        """

        # Extract the cloud cover probability for the month.
        cloud_cover_prob = self._monthly_weather_data[date_and_time.month].cloud_cover

        # Generate a random number between 1 and 0 for that month based on this factor.
        rand_prob: float = random.randrange(0, RAND_RESOLUTION, 1) / RAND_RESOLUTION

        # Determine what effect the cloudy (or not cloudy) conditions have on the solar
        # insolation. Generate a fractional reduction based on this.
        # Return this number
        return cloud_cover_prob * rand_prob * cloud_efficacy_factor

    def _ambient_temperature(
        self, latitude: float, longitude: float, date_and_time: datetime.datetime
    ) -> float:
        """
        Return the ambient temperature, in Kelvin, based on the date and time.

        A sine curve is fitted, and the temp extracted.

        The models used in this function are obtained, with permission, from
        https://mathscinotes.com/wp-content/uploads/2012/12/dailytempvariation.pdf
        and use an improved theoretical model from a previous paper.

        :param latitude:
            The latitude of the set-up.

        :param longitude:
            The longitude of the set-up.

        :param date_and_time:
            The current date and time.

        :return:
            The temperature in Kelvin.

        """

        max_temp = self._monthly_weather_data[date_and_time.month].day_temp
        min_temp = self._monthly_weather_data[date_and_time.month].night_temp
        temp_range = max_temp - min_temp

        if (
            self._monthly_weather_data[date_and_time.month].sunrise is None
            or self._monthly_weather_data[date_and_time.month].sunset is None
        ):
            self._monthly_weather_data[date_and_time.month].sunrise = _get_sunrise(
                latitude, longitude, date_and_time.replace(hour=0)
            )
            self._monthly_weather_data[date_and_time.month].sunset = _get_sunset(
                latitude, longitude, date_and_time.replace(hour=23)
            )

        return (
            temp_range
            * math.exp(-(date_and_time.hour + date_and_time.minute / 60 - 12) * G_CONST)
            * (1 + (date_and_time.hour + date_and_time.minute / 60 - 12) / 12)
            ** (G_CONST * 12)
            + min_temp
        )

    def get_weather(
        self,
        latitude: float,
        longitude: float,
        cloud_efficacy_factor: float,
        date_and_time: datetime.datetime,
    ) -> WeatherConditions:
        """
        Computes the solar irradiance based on weather conditions at the time of day.

        :param latitude:
            The latitude of the PV-T set-up.

        :param longitude:
            The longitude of the PV-T set-up.

        :param cloud_efficacy_factor:
            The extend to which cloud cover affects the sunlight. This is multiplied by
            a random number which further reduces the effect of the cloud cover in
            reducing the sunlight.

        :param date_and_time:
            The date and time of day, used to calculate the irradiance.

        :return:
            A :class:`__utils__.WeatherConditions` giving the solar irradiance, in
            watts per meter squared, and the angles, both azimuthal and declination, of
            the sun's position in the sky.

        """

        # Based on the time, compute the sun's position in the sky, making sure to
        # account for the seasonal variation.
        azimuthal_angle, declination = _get_solar_angles(
            latitude, longitude, date_and_time
        )

        # Factor in the weather conditions and cloud cover to compute the current solar
        # irradiance.
        irradiance: float = (
            self._solar_insolation
            * (1 - self._cloud_cover(cloud_efficacy_factor, date_and_time))
            if declination > 0
            else 0
        )

        # * Compute the wind speed
        wind_speed: float = 0

        # Compute the ambient temperature.
        ambient_temperature = self._ambient_temperature(
            latitude, longitude, date_and_time
        )

        # Return all of these in a WeatherConditions variable.
        return WeatherConditions(
            irradiance, declination, azimuthal_angle, wind_speed, ambient_temperature
        )
