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
import collections
import datetime
import logging
import math
import random

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import pysolar

from .__utils__ import (
    ZERO_CELCIUS_OFFSET,
    LOGGER_NAME,
    MissingParametersError,
    ProgrammerJudgementFault,
    BaseDailyProfile,
    Date,
    WeatherConditions,
    read_yaml,
)

__all__ = ("WeatherForecaster",)


# A parameter used in modelling weather data curves.
G_CONST = 1

# The resolution to which random numbers are generated
RAND_RESOLUTION = 100

logger = logging.getLogger(LOGGER_NAME)


class _DailySolarIrradianceProfile(BaseDailyProfile):
    """
    Represents a day's solar irradiance profile.

    """

    @property
    def profile(self) -> Dict[datetime.time, float]:
        """
        Publically yields the internal "_profile" for average purposes.

        :return:
            The profile as a mapping between time and value.

        """

        return self._profile


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

    .. attribute:: solar_irradiance_profiles
        A mapping between the day in the month and the solar irradiance profile for
        that day.

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
    solar_irradiance_profile: Optional[Dict[int, _DailySolarIrradianceProfile]] = None
    _average_irradiance_profile: Optional[_DailySolarIrradianceProfile] = None
    sunrise: Optional[datetime.time] = None
    sunset: Optional[datetime.time] = None

    def __repr__(self) -> str:
        """
        Return a standard representation of the class.

        :return:
            A nicely-formatted monthly-weather data string.

        """

        return (
            "_MonthlyWeatherData(month: {}, num_days: {}, cloud_cover: {}, ".format(
                self.month_name,
                self.num_days,
                self.cloud_cover,
            )
            + "rainy_days: {}, day_temp: {}, night_temp: {}, ".format(
                self.rainy_days,
                self.day_temp,
                self.night_temp,
            )
            + "solar_profiles_set: [{}])".format(
                "X" if self.solar_irradiance_profile is not None else " "
            )
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

    @property
    def average_irradiance_profile(self) -> _DailySolarIrradianceProfile:
        """
        Returns the average daily solar irradiance profile for the month.

        :return:
            The average solar irradiance profile for the month as a
            :class:`_DailySolarIrradianceProfile` instance.

        """

        if self._average_irradiance_profile is not None:
            return self._average_irradiance_profile

        average_irradiance_profile_counter = collections.Counter(dict())

        for profile in self.solar_irradiance_profile.values():
            average_irradiance_profile_counter += collections.Counter(profile.profile)

        self._average_irradiance_profile = _DailySolarIrradianceProfile(
            {
                key: value / len(average_irradiance_profile_counter)
                for key, value in dict(average_irradiance_profile_counter).items()
            }
        )

        return self._average_irradiance_profile


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
    # .. attribute:: _average_irradiance
    #   A `bool` giving whether to average the solar irradiance intensities (True), or
    #   use the data for each day as called (False).
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
        average_irradiance: bool,
        mains_water_temp: float,
        monthly_weather_data: Dict[str, Dict[str, Union[str, float]]],
        solar_irradiance_data: Dict[int, Dict[int, Dict[datetime.time, float]]],
    ) -> None:
        """
        Instantiate a weather forecaster class.

        :param average_irradiance:
            Whether to average the solar intensity for each month (True), or use the
            data for each day (False) as required.

        :param mains_water_temp:
            The mains water temperature, measured in Kelvin.

        :param monthly_weather_data:
            The monthly weather data, extracted raw from the weather data YAML file.

        :param solar_irradiance_data:
            A mapping between :class:`__utils__.Date` instances and irradiance profiles.

        """

        self._average_irradiance = average_irradiance

        self.mains_water_temp = mains_water_temp + ZERO_CELCIUS_OFFSET

        self._monthly_weather_data = {
            self._month_abbr_to_num[month]: _MonthlyWeatherData.from_yaml(
                month, month_data  # type: ignore
            )
            for month, month_data in monthly_weather_data.items()
        }

        for month, weather_data in self._monthly_weather_data.items():
            weather_data.solar_irradiance_profile = {
                date.day: profile
                for date, profile in solar_irradiance_data.items()
                if date.month == month
            }

    def __repr__(self) -> str:
        """
        Return a nice-looking representation of the weather forecaster.

        :return:
            A nicely-formatted string giving information about the weather forecaster.

        """

        return "WeatherForecaster( mains_water_temp: {}, ".format(
            self.mains_water_temp
        ) + "num_months: {})".format(len(self._monthly_weather_data.keys()))

    @classmethod
    def from_data(
        cls,
        average_irradiance: bool,
        weather_data_path: str,
        solar_irradiance_filenames: List[str],
    ) -> Any:
        """
        Instantiate a :class:`WeatherForecaster` from paths to various data files.

        :param average_irradiance:
            Whether to average the solar irradiances wihtin each month (True), or use
            the value for each day specifically when called (False).

        :param weather_data_path:
            The path to the weather-data file. This contains basic information about the
            location that is weather-related.

        :param solar_irradiance_filenames:
            A `list` of paths to files containing solar irradiance profiles. The format
            of these files is in the form as given by the PVGIS (Photovoltaic,
            Geographic Information Service). Processing of these files occurs here.

        :return:
            A :class:`WeatherForecaster` instance.

        """

        # Call out to the __utils__ module to read the yaml data.
        data = read_yaml(weather_data_path)

        # * Check that all months are specified.
        try:
            data.pop("solar_insolation")
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

        # Extract the solar irradiance data from the files.
        solar_irradiance_data: Dict[
            Date : Dict[datetime.time, float]
        ] = collections.defaultdict(_DailySolarIrradianceProfile)
        # Extract and open each file in series.
        for filename in solar_irradiance_filenames:
            with open(filename) as f:
                filedata = json.load(f)
                # Compile a dictionary consisting of the combined data from all files.
            processed_filedata = {
                datetime.datetime.strptime(entry["time"], "%Y%m%d:%H%M"): entry["G(i)"]
                for entry in filedata["outputs"]["hourly"]
            }

            # Create a temporary mapping of date to daily profile.
            date_to_profile_mapping: Dict[Date:Dict] = collections.defaultdict(dict)
            for date_and_time, irradiance in processed_filedata.items():
                date_to_profile_mapping[Date.from_date(date_and_time.date())].update(
                    {date_and_time.time(): irradiance}
                )

            # Update the running dictionary.
            for date, profile in date_to_profile_mapping.items():
                solar_irradiance_data[date].update(profile)

        # Instantiate and return a Weather Forecaster based off of this weather data.
        return cls(average_irradiance, mains_water_temp, data, solar_irradiance_data)

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

    def _irradiance(self, date_and_time: datetime.datetime) -> float:
        """
        Return the solar irradiance, in Watts per meter squared, at the current date and
        time.

        :param date_and_time:
            The date and time.

        :return:
            The solar irradiance, in Watts per meter squared.

        """

        if self._average_irradiance:
            return self._monthly_weather_data[
                date_and_time.month
            ].average_irradiance_profile[date_and_time.time()]

        return self._monthly_weather_data[date_and_time.month].solar_irradiance_profile[
            date_and_time.day
        ][date_and_time.time()]

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
        irradiance: float = self._irradiance(date_and_time) * (
            1 - self._cloud_cover(cloud_efficacy_factor, date_and_time)
        )

        # * Compute the wind speed
        wind_speed: float = 5  # [m/s]

        # Compute the ambient temperature.
        ambient_temperature = self._ambient_temperature(
            latitude, longitude, date_and_time
        )

        # Return all of these in a WeatherConditions variable.
        return WeatherConditions(
            irradiance, declination, azimuthal_angle, wind_speed, ambient_temperature
        )
