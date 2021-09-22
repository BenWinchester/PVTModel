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
import os
import random

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple, Union

import json
import pysolar

from ..__utils__ import MissingParametersError, read_yaml


from .constants import (
    ZERO_CELCIUS_OFFSET,
)
from .__utils__ import (
    BaseDailyProfile,
    Date,
    ProgrammerJudgementFault,
    PVT_SYSTEM_MODEL_LOGGER_NAME,
    WeatherConditions,
)

__all__ = ("WeatherForecaster",)


# A parameter used in modelling weather data curves.
G_CONST = 1

# The resolution to which random numbers are generated
RAND_RESOLUTION = 100

logger = logging.getLogger(PVT_SYSTEM_MODEL_LOGGER_NAME)


class _DailyProfile(BaseDailyProfile):
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

    def __add__(self, other) -> Any:  # type: ignore
        """
        Adds two profiles together and returns the result.

        :param other:
            The other instance of a :class:`_DailySolarIrradianceProilfe` to which this
            one should be add, then the two returned.

        :return:
            A new :class:`_DailyProfile` instance, instantiated with the
            profile made by combining the two profiles.

        """

        profile = {
            key: self_value + other.profile[key]
            for key, self_value in self.profile.items()
        }

        return _DailyProfile(profile)

    def __truediv__(self, divisor: float) -> Any:  # type: ignore
        """
        Divides every value in the profile by the divisor.

        :param divisor:
            The number by which every value in the profile should be divided.

        :return:
            A new :class:`_DailyProfile` instance, instantiated with the
            profile made by dividing the current profile by the divisor.

        """

        profile = {key: value / divisor for key, value in self.profile.items()}

        return _DailyProfile(profile)


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

    .. attribute:: override_irradiance_profile
        If set, this will override the default behaviour of searching for the internal
        profile for the given day, or computing the average irradiance profile, and will
        simply return the profile set on this attribute when called.

    .. attribute:: average_temperature_profile
        The average temperature profile for the month.

    .. attribute:: sunrise
        The sunrise time. Can be set later.

    .. attribute:: sunset
        The sunset time. Can be set later.

    """

    # Private Attributes:
    #
    # .. attribute:: _average_irradiance_profile
    #   Used to store the average solar irradiance profile for the month.
    #

    month_name: str
    num_days: float
    cloud_cover: float
    rainy_days: float
    day_temp: float
    night_temp: float
    solar_irradiance_profiles: Dict[int, _DailyProfile] = field(default_factory=dict)
    _average_irradiance_profile: _DailyProfile = _DailyProfile(dict())
    override_irradiance_profile: _DailyProfile = _DailyProfile(dict())
    average_temperature_profile: _DailyProfile = _DailyProfile(dict())
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
                "X" if self.solar_irradiance_profiles is not None else " "
            )
        )

    @classmethod
    def from_yaml(
        cls, month_name: str, monthly_weather_data: Dict[str, Union[str, float]]
    ) -> Any:  # type: ignore
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
    def average_irradiance_profile(self) -> _DailyProfile:
        """
        Returns the average daily solar irradiance profile for the month.

        :return:
            The average solar irradiance profile for the month as a
            :class:`_DailyProfile` instance.

        """

        if self.override_irradiance_profile is not None:
            return self.override_irradiance_profile

        if self._average_irradiance_profile is not None:
            return self._average_irradiance_profile

        # Instantiate a counter.
        counter: Dict[datetime.time, float] = collections.defaultdict(float)

        if self.solar_irradiance_profiles is None:
            raise ProgrammerJudgementFault(
                "The irradiance profile in the weather module should not be None."
            )

        # Loop through all internally-held profiles, and sum up the values at each given
        # time, dividing by the number of days in the month.
        for profile in self.solar_irradiance_profiles.values():
            for time, value in profile.profile.items():
                counter[time] += value / self.num_days

        # Store this as a profile on the class for future use, and return this profile.
        self._average_irradiance_profile = _DailyProfile(counter)

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

    .. attribute:: mains_water_temperature
        The mains water temperature, measured in Kelvin.

    """

    # Private attributes:
    #
    # .. attribute:: _average_irradiance
    #   A `bool` giving whether to average the solar irradiance intensities (True), or
    #   use the data for each day as called (False).
    #
    # .. attribute:: _average_air_pressure
    #   The average air pressure measured in Pascals.
    #
    # .. attribute:: _month_abbr_to_num
    #   A mapping from month abbreviated name (eg, "jan", "feb" etc) to the number of
    #   the month in the year.
    #
    # .. attribute:: _monthly_weather_data
    #   A `dict` mapping month number to :class:`_MonthlyWeatherData` instances
    #   containing weather information for that month.
    #
    # .. attribute:: _override_ambient_temperature
    #   The ambient temperature, measured in Kelvin, used as a constant value rather
    #   than time-s[ecific data.
    #
    # .. attribute:: _override_irradiance
    #   The solar irradiance, measured in Watts per meter squared, to use as a constant
    #   value rather than time-specific data.
    #
    # .. attribute:: _override_wind_speed
    #   The wind speed, measured in meters per second, used as a constant value rather
    #   than time-specific data.
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
        ambient_tank_temperature: float,
        average_air_pressure: float,
        average_irradiance: bool,
        mains_water_temperature: float,
        monthly_weather_data: Dict[str, Dict[str, Union[str, float]]],
        monthly_irradiance_profiles: Dict[Date, _DailyProfile],
        monthly_temperature_profiles: Dict[Date, _DailyProfile],
        override_ambient_temperature: Optional[float],
        override_irradiance: Optional[float],
        overridw_wind_speed: Optional[float],
    ) -> None:
        """
        Instantiate a weather forecaster class.

        :param ambient_tank_temperature:
            The average ambient temperature surrounding the hot-water wank.

        :param average_air_pressure:
            The average air pressure, measured in Pascals.

        :param average_irradiance:
            Whether to average the solar intensity for each month (True), or use the
            data for each day (False) as required.

        :param mains_water_temperature:
            The mains water temperature, measured in Kelvin.

        :param monthly_weather_data:
            The monthly weather data, extracted raw from the weather data YAML file.

        :param monthly_irradiance_profiles:
            A mapping between :class:`__utils__.Date` instances and solar irradiance
            profiles stored as :class:`_DailyProfile` instances.

        :param override_ambient_temperature:
            Used to override the ambient temperature profiles with a constant value.

        :param override_irradiance:
            Used to override the solar-irradiance profiles with a constant value.

        :param override_wind_speed:
            Used to override the wind-speed profiles with a constant value.

        """

        self._average_air_pressure = average_air_pressure
        self._average_irradiance = average_irradiance
        self._override_ambient_temperature = override_ambient_temperature
        self._override_irradiance = override_irradiance
        self._override_wind_speed = overridw_wind_speed
        self.ambient_tank_temperature = ambient_tank_temperature
        self.mains_water_temperature = mains_water_temperature + ZERO_CELCIUS_OFFSET

        self._monthly_weather_data: Dict[int, _MonthlyWeatherData] = {
            self._month_abbr_to_num[month]: _MonthlyWeatherData.from_yaml(
                month, month_data  # type: ignore
            )
            for month, month_data in monthly_weather_data.items()
        }

        for month, weather_data in self._monthly_weather_data.items():
            weather_data.solar_irradiance_profiles = {
                date.day: profile
                for date, profile in monthly_irradiance_profiles.items()
                if date.month == month
            }
            if len(weather_data.solar_irradiance_profiles) == 1:
                weather_data.override_irradiance_profile = next(
                    iter(weather_data.solar_irradiance_profiles.values())
                )
            try:
                weather_data.average_temperature_profile = monthly_temperature_profiles[
                    Date(1, month)
                ]
            except KeyError as e:
                logger.debug(
                    "Unable to set average temperature profile for month %s; "
                    "data not present: %s",
                    month,
                    str(e),
                )
                continue

    def __repr__(self) -> str:
        """
        Return a nice-looking representation of the weather forecaster.

        :return:
            A nicely-formatted string giving information about the weather forecaster.

        """

        return (
            "WeatherForecaster("
            f"_average_irradiance: {self._average_irradiance}, "
            f"mains_water_temperature: {self.mains_water_temperature}K, "
            f"num_months: {len(self._monthly_weather_data.keys())}, "
            f"override_ambient_temperature: {self._override_ambient_temperature}K, "
            f"override_irradiance: {self._override_irradiance}W/m^2, "
            f"overridw_wind_speed: {self._override_wind_speed}m/s"
            ")"
        )

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

        return self._monthly_weather_data[
            date_and_time.month
        ].solar_irradiance_profiles[date_and_time.day][date_and_time.time()]

    @classmethod
    def from_data(  # pylint: disable=too-many-branches
        cls,
        average_irradiance: bool,
        solar_irradiance_filenames: Set[str],
        temperature_filenames: Set[str],
        weather_data_path: str,
        override_ambient_temperature: Optional[float] = None,
        override_irradiance: Optional[float] = None,
        override_wind_speed: Optional[float] = None,
        use_pvgis: bool = False,
    ) -> Any:  # type: ignore
        """
        Instantiate a :class:`WeatherForecaster` from paths to various data files.

        :param average_irradiance:
            Whether to average the solar irradiances wihtin each month (True), or use
            the value for each day specifically when called (False).

        :param solar_irradiance_filenames:
            A `set` of paths to files containing solar irradiance profiles. The format
            of these files is in the form as given by the PVGIS (Photovoltaic,
            Geographic Information Service). Processing of these files occurs here.

        :param temperature_filenames:
            A `set` of paths to files containing temperature profiles.

        :param weather_data_path:
            The path to the weather-data file. This contains basic information about the
            location that is weather-related.

        :param override_ambient_temperature:
            If specified, the value will be used to override the internal ambient
            temperature profile(s). The value should be in degrees Kelvin.

        :param override_irradiance:
            If specified, the value will be used to override the internal solar
            irradiance profile(s).

        :param override_wind_speed:
            If specified, the value will be used to override the internal wind-speed
            profile(s).

        :param use_pvgis:
            Whether to use data obtained from PVGIS (True) or not (False).

        :return:
            A :class:`WeatherForecaster` instance.

        """

        data = read_yaml(weather_data_path)

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

        try:
            ambient_tank_temperature = (
                data.pop("average_ambient_household_temperature") + ZERO_CELCIUS_OFFSET
            )
        except KeyError:
            logger.error(
                "Weather forecaster from %s is missing "
                "'average_ambient_household_temperature' data.",
                weather_data_path,
            )
            raise MissingParametersError(
                "WeatherForecaster",
                "The average ambient tank temperature param is missing from {}.".format(
                    weather_data_path
                ),
            ) from None

        try:
            average_air_pressure = data.pop("average_air_pressure")
        except KeyError:
            logger.error(
                "Weather forecaster from %s is missing " "'average_air_pressure' data.",
                weather_data_path,
            )
            raise MissingParametersError(
                "WeatherForecaster",
                "The average air pressure param is missing from {}.".format(
                    weather_data_path
                ),
            ) from None

        # NOTE:
        # > Here, PVGIS code exists in a block s.t. the override profiles obtained from
        # > Maria's paper using a highly scientific method :p can be used instead.

        # Loop through all data files, reading in the data, and adding to a profile
        # keyed only by month and day.
        if use_pvgis:
            temp_irradiance_profiles: Dict[Date, dict] = collections.defaultdict(dict)

            for filename in solar_irradiance_filenames:
                with open(filename) as f:
                    filedata = json.load(f)
                processed_filedata: Dict[datetime.datetime, float] = {
                    datetime.datetime.strptime(entry["time"], "%Y%m%d:%H%M"): entry[
                        "G(i)"
                    ]
                    for entry in filedata["outputs"]["hourly"]
                }

                num_years = len({date.year for date in processed_filedata})

                for key, value in processed_filedata.items():
                    # This is being wrapped in a try-except block to allow for assigning of
                    # both items that are already present, and those which aren't.
                    try:
                        temp_irradiance_profiles[Date.from_date(key.date())][
                            key.time()
                        ] += (value / num_years)
                    except KeyError:
                        temp_irradiance_profiles[Date.from_date(key.date())][
                            key.time()
                        ] = (value / num_years)

            # These profiles now need to be cast to _DailyProfiles
            monthly_irradiance_profiles: Dict[Date, _DailyProfile] = {
                date: _DailyProfile(profile)
                for date, profile in temp_irradiance_profiles.items()
            }
        else:
            # Cycle through the various profile files, opening the profiles and storing as a
            # mapping.
            monthly_irradiance_profiles = dict()
            for filename in solar_irradiance_filenames:
                with open(filename, "r") as f:
                    filedata = json.load(f)

                # Process the profile and store it.
                try:
                    monthly_irradiance_profiles[
                        Date(1, cls._month_abbr_to_num[os.path.basename(filename)[:3]])
                    ] = _DailyProfile(
                        {
                            datetime.datetime.strptime(key, "%H:%M").time(): value
                            for key, value in filedata.items()
                        }
                    )
                except ValueError:
                    logger.info(
                        "Weather data contains additional information. Attempting to use convert."
                    )
                    monthly_irradiance_profiles[
                        Date(1, cls._month_abbr_to_num[os.path.basename(filename)[:3]])
                    ] = _DailyProfile(
                        {
                            datetime.datetime.strptime(key, "%H:%M:%S").time(): value
                            for key, value in filedata.items()
                        }
                    )

        temperature_profiles: Dict[Date, _DailyProfile] = dict()
        for filename in temperature_filenames:
            with open(filename, "r") as f:
                filedata = json.load(f)

            # Process the profile and store it.
            try:
                temperature_profiles[
                    Date(1, cls._month_abbr_to_num[os.path.basename(filename)[:3]])
                ] = _DailyProfile(
                    {
                        datetime.datetime.strptime(key, "%H:%M").time(): value
                        + ZERO_CELCIUS_OFFSET
                        for key, value in filedata.items()
                    }
                )
            except ValueError:
                logger.info(
                    "Weather data contains additional information. Attempting to use convert."
                )
                temperature_profiles[
                    Date(1, cls._month_abbr_to_num[os.path.basename(filename)[:3]])
                ] = _DailyProfile(
                    {
                        datetime.datetime.strptime(key, "%H:%M:%S").time(): value
                        + ZERO_CELCIUS_OFFSET
                        for key, value in filedata.items()
                    }
                )

        # # @@@ For now, the solar irradiance profiles and temperature profiles for the
        # # @@@ missing months are filled in here.
        # monthly_irradiance_profiles[Date(1, 2)] = monthly_irradiance_profiles[
        #     Date(1, 1)
        # ]
        # monthly_irradiance_profiles[Date(1, 3)] = monthly_irradiance_profiles[
        #     Date(1, 4)
        # ]
        # monthly_irradiance_profiles[Date(1, 5)] = monthly_irradiance_profiles[
        #     Date(1, 4)
        # ]
        # monthly_irradiance_profiles[Date(1, 6)] = monthly_irradiance_profiles[
        #     Date(1, 8)
        # ]
        # monthly_irradiance_profiles[Date(1, 7)] = _DailyProfile(
        #     {
        #         key: (
        #             monthly_irradiance_profiles[Date(1, 8)].profile[key] * 3
        #             + monthly_irradiance_profiles[Date(1, 4)].profile[key]
        #         )
        #         / 4
        #         for key in monthly_irradiance_profiles[Date(1, 8)].profile.keys()
        #     }
        # )
        # monthly_irradiance_profiles[Date(1, 9)] = monthly_irradiance_profiles[
        #     Date(1, 4)
        # ]
        # monthly_irradiance_profiles[Date(1, 10)] = monthly_irradiance_profiles[
        #     Date(1, 4)
        # ]
        # monthly_irradiance_profiles[Date(1, 11)] = monthly_irradiance_profiles[
        #     Date(1, 1)
        # ]
        # monthly_irradiance_profiles[Date(1, 12)] = monthly_irradiance_profiles[
        #     Date(1, 1)
        # ]

        # temperature_profiles[Date(1, 2)] = temperature_profiles[Date(1, 1)]
        # temperature_profiles[Date(1, 3)] = temperature_profiles[Date(1, 4)]
        # temperature_profiles[Date(1, 5)] = temperature_profiles[Date(1, 4)]
        # temperature_profiles[Date(1, 6)] = temperature_profiles[Date(1, 7)]
        # temperature_profiles[Date(1, 9)] = temperature_profiles[Date(1, 4)]
        # temperature_profiles[Date(1, 10)] = temperature_profiles[Date(1, 4)]
        # temperature_profiles[Date(1, 11)] = temperature_profiles[Date(1, 1)]
        # temperature_profiles[Date(1, 12)] = temperature_profiles[Date(1, 1)]

        # Instantiate and return a Weather Forecaster based off of this weather data.
        return cls(
            ambient_tank_temperature,
            average_air_pressure,
            average_irradiance,
            mains_water_temp,
            data,
            monthly_irradiance_profiles,
            temperature_profiles,
            override_ambient_temperature + ZERO_CELCIUS_OFFSET
            if override_ambient_temperature is not None
            else None,
            override_irradiance,
            override_wind_speed,
        )

    def get_weather(
        self,
        latitude: float,
        longitude: float,
        cloud_efficacy_factor: float,  # pylint: disable=unused-argument
        date_and_time: Optional[datetime.datetime] = None,
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

        # Factor in the weather conditions and cloud cover to compute the current solar
        # irradiance.
        if date_and_time is not None:
            # Based on the time, compute the sun's position in the sky, making sure to
            # account for the seasonal variation.
            azimuthal_angle, declination = _get_solar_angles(
                latitude, longitude, date_and_time
            )
            irradiance: float = self._irradiance(date_and_time)  # * (
            # 1 - self._cloud_cover(cloud_efficacy_factor, date_and_time)
            # )
            # >>> The ambient temperature is now determined from a temperature profile.
            # # Compute the ambient temperature.
            # ambient_temperature: float = self._ambient_temperature(
            #     latitude, longitude, date_and_time
            # )
            # <<< >>> Ambient temperature determination using temperature profile.
            ambient_temperature: float = self._monthly_weather_data[
                date_and_time.month
            ].average_temperature_profile[date_and_time.time()]
            # <<< E.O. code-alteration block
        else:
            azimuthal_angle = 180
            declination = 45
            if self._override_irradiance is None:
                raise MissingParametersError(
                    "WeatherForecaster",
                    "The weather forecaster was called for time-independant weather "
                    "data without an override irradiance provided.",
                )
            irradiance = self._override_irradiance
            if self._override_ambient_temperature is None:
                raise MissingParametersError(
                    "WeatherForecaster",
                    "The weather forecaster was called for time-independant weather "
                    "data without an override ambient temperature provided.",
                )
            ambient_temperature = self._override_ambient_temperature

        # * Compute the wind speed
        if self._override_wind_speed is not None:
            wind_speed: float = self._override_wind_speed  # [m/s]
        else:
            wind_speed = 5

        # Return all of these in a WeatherConditions variable.
        return WeatherConditions(
            _irradiance=irradiance,
            azimuthal_angle=azimuthal_angle,
            ambient_tank_temperature=self.ambient_tank_temperature,
            ambient_temperature=ambient_temperature,
            declination=declination,
            mains_water_temperature=self.mains_water_temperature,
            pressure=self._average_air_pressure,
            wind_speed=wind_speed,
        )
