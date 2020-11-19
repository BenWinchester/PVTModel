#!/usr/bin/python3.7
########################################################################################
# load.py - Computs load characteristics.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The load module for this PV-T model.

This module computes the load at any given instance based on various probabilistic
factors read in from a data file and stored within the module when running.

"""

import collections
import datetime
import enum

from dataclasses import dataclass
from typing import Any, Dict

from .__utils__ import (
    InternalError,
    InvalidDataError,
    MissingDataError,
    MissingParametersError,
    read_yaml,
)

__all__ = (
    "LoadData",
    "LoadProfile",
    "LoadSystem",
)


class _Season(enum.Enum):
    """
    Used to label the season.

    """

    SEASONLESS = 0
    SPRING = 1
    SUMMER = 2
    AUTUMN = 3
    WINTER = 4

    @classmethod
    def from_month(cls, month: int) -> Any:
        """
        Return the season based on the month.

        :param month:
            The month.

        :return:
            A :class:`_Season` corresponding to the given month.

        """

        try:
            return cls((month % 12 + 3) // 3)
        except TypeError as e:
            raise InvalidDataError(
                "N/A",
                "The input '{}' must be an integer when determining seasons: {}".format(
                    month, str(e)
                ),
            ) from None


class _Day(enum.Enum):
    """
    Used to label the type of day, eg, weekday, Saturday, or Sunday.

    """

    DAYLESS = 0
    WEEKDAY = 1
    SATURDAY = 2
    SUNDAY = 3


class _ProfileType(enum.Enum):
    """
    Used to label whether a profile is a hot-water or electrical (or other) profile.

    """

    ELECTRICAL = 0
    HOT_WATER = 1


@dataclass
class LoadData:
    """
    Conaints information about the various loads on the system.

    .. attribute:: electrical_load
        The electrical load on the system, measured in Watts.

    .. attribute:: hot_water_load
        The hot-water load on the system, measured in Litres.

    """

    electrical_load: float
    hot_water_load: float


# class Device:
#     """
#     Represents a single device that can be conected, either electrical or thermal.

#     .. attribute:: name
#         A `str` giving a label with which the device can be referenced.

#     """

#     # Private Attributes:
#     #
#     # .. attribute:: _load
#     #   The device load, when used, measured in Watts.
#     #
#     # .. attribute:: _weekday_daytime_usage_probability
#     #   The probability that the device is used during the daytime during the week.
#     #
#     # .. attribute:: _weekend_daytime_usage_probability
#     #   The probability that the device is used during the daytime over the weekend.
#     #
#     # .. attribute:: _weekday_nighttime_usage_probability
#     #   The probability that the device is used during the night during the week.
#     #
#     # .. attribute:: _weekend_nighttime_usage_probability
#     #   The probability that the device is used during the night over the weekend.
#     #

#     def __init__(
#         self,
#         name: str,
#         load: float,
#         weekday_daytime_usage_probability: float,
#         weekend_daytime_usage_probability: float,
#         weekday_nighttime_usage_probability: float,
#         weekend_nighttime_usage_probability: float,
#     ) -> None:
#         """
#         Instantiate a :class:`Device` instance.

#         :param name:
#             A label to assign as the name of the device.

#         :param load:
#             The load of the device, when used, measured in Watts.

#         :param weekday_daytime_usage_probability:
#             The probability that the device is used during the daytime during the week.

#         :param weekend_daytime_usage_probability:
#             The probability that the device is used during the daytime over the weekend.

#         :param weekday_nighttime_usage_probability:
#             The probability that the device is used during the night during the week.

#         :param weekend_nighttime_usage_probability:
#             The probability that the device is used during the night over the weekend.

#         """

#         self.name = name
#         self._load = load
#         self._weekday_daytime_usage_probability = weekday_daytime_usage_probability
#         self._weekend_daytime_usage_probability = weekend_daytime_usage_probability
#         self._weekday_nighttime_usage_probability = weekday_nighttime_usage_probability
#         self._weekend_nighttime_usage_probability = weekend_nighttime_usage_probability

#     @classmethod
#     def from_yaml(cls, device_entry: Dict[Any, Any]) -> Any:
#         """
#         Returns a :class:`Device` instance from yaml data in the form of a dictionary.

#         :param device_entry:
#             A `dict` representing the device based on raw yaml data.

#         :return:
#             A :class:`Device` instance based on the data.

#         """

#         # Check that all the required fields are present.

#         # ? Maybe this checking should be done when the YAML data is read in?

#         # * Generate, and return, a Device instance as appropriate.

#         return cls("", 0, 0, 0, 0, 0)

#     def load(self, date_and_time: datetime.datetime) -> float:
#         """
#         Returns the current load of the device, in Watts, based on the date and time.

#         :param date_and_date:
#             The current date and time.

#         :return:
#             The load of the device in Watts based on the probability of it being used.

#         """

#         # * Determine whether it is a weekend or weekday and whether it's day or night.

#         # * Determine the instantaneous load of the device and return it.


class LoadProfile:
    """
    Represents a load profile.

    .. attribute:: resolution
        The time-step resolution of the profile, measured in minues.

    """

    # Private Attributes:
    #
    #  .. attribute:: _profile
    #   A mapping of hour to the load at that hour.
    #

    def __init__(self, resolution: int, profile_data: Dict[str, float]) -> None:
        """
        Instantiate a load profile.

        :param resolution:
            The resolution of the load profile, measured in minutes.

        :param profile_data:
            The profile data: a mapping of time to the load at that time step.

        """

        self.resolution = resolution
        self._profile = profile_data

    @classmethod
    def from_yaml(cls, yaml_data: Dict[Any, Any]) -> Any:
        """
        Instantiate a load profile based on some YAML data.

        :param yaml_data:
            The raw, load-profile data extracted from a YAML file.

        :return:
            A :class:`LoadProfile` instance, based on the YAML data provided.

        """

        # Convert the data entries to floating point numbers.
        try:
            yaml_data = {str(key): float(value) for key, value in yaml_data.items()}
        except ValueError as e:
            raise InternalError(str(e)) from None

        try:
            resolution = list(yaml_data.keys())[1] - list(yaml_data.keys())[0]
        except TypeError as e:
            raise InternalError(str(e)) from None

        return cls(resolution, yaml_data)

    def load(self, time: datetime.time) -> float:
        """
        Get the load from the load profile at the given time.

        :param resolution:
            The time-step resolution (in minutes) of the model being run.

        :param time:
            The time for which to fetch the load data.

        :return:
            The load at this given time.

        """

        try:
            return self._profile["{:02d}{:02d}".format(time.hour, time.minute)]
        except KeyError as e:
            raise MissingDataError(
                "A request was made for load data at time "
                "{:02d}:{:02d}. This was undefined: {}".format(
                    time.hour, time.minute, str(e)
                )
            ) from None


class LoadSystem:
    """
    Represents a system, eg, a household, potential of generating loads.

    """

    # Private Attributes:
    #
    # .. attribute:: _seasonal_load_profiles
    #   A mapping between the profile type, season, and day of the profile.
    #

    def __init__(
        self,
        seasonal_load_profiles: Dict[
            _ProfileType : Dict[_Season : Dict[_Day:LoadProfile]]
        ],
    ) -> None:
        """
        Instantiate a Load System.

        :param seasonal_load_profiles:
            A mapping representing the internal structure of how load profiles are to be
            stored.

        """

        self._seasonal_load_profiles = seasonal_load_profiles

    @classmethod
    def from_yaml(cls, load_data_path: str) -> Any:
        """
        Returns a :class:`LoadSystem` instance based on the YAML data inputted.

        """

        yaml_data = read_yaml(load_data_path)
        seasonal_load_profiles: collections.defaultdict = collections.defaultdict(dict)

        for profile_type, type_data in yaml_data.values():
            for season, seasonal_data in type_data.values():
                for day, day_data in seasonal_data.values():
                    try:
                        seasonal_load_profiles[
                            _ProfileType.__members__[profile_type.upper()]
                        ][_Season.__members__[season.upper()]][
                            _Day.__members__[day.upper()]
                        ] = LoadProfile.from_yaml(
                            day_data
                        )
                    except InternalError as e:
                        raise InvalidDataError(
                            load_data_path,
                            "Error processing Load YAML data for entry: "
                            "{}/{}/{}: {}".format(profile_type, season, day, str(e)),
                        ) from None

        return cls(seasonal_load_profiles)

    def get_load_from_time(
        self, load_data: LoadData, date_and_time: datetime.datetime
    ) -> LoadData:
        """
        Return the electrical and hot-water loads based on YAML data at a given time.

        :param load_data:
            The type of load data being requested.

        :param date_and_time:
            The date and time at which the load data should be fetched.

        :return:
            The hot-water and electrical data.

        """

        # Attempt to extract season-specific data.
        try:
            return self._seasonal_load_profiles[load_data][
                _Season.from_month(date_and_time.month)
            ][_Day(date_and_time.weekday())].load(date_and_time.time())
        except KeyError:
            pass

        # Try to extract day-unspecific data.
        try:
            return self._seasonal_load_profiles[load_data][
                _Season.from_month(date_and_time.month)
            ][_Day(0)].load(date_and_time.time())
        except KeyError:
            pass

        # Try to extract season-unspecific data.
        try:
            return self._seasonal_load_profiles[load_data][_Season(0)][_Day(0)].load(
                date_and_time.time()
            )
        except KeyError as e:
            raise MissingDataError(
                "Load data could not be obtained. "
                "Attempt was for {} data ".format(str(load_data))
                + "on {:02d}/{:02d}/{:04d} at time {:02d}:{:02d}: {}".format(
                    date_and_time.day,
                    date_and_time.month,
                    date_and_time.year,
                    date_and_time.hour,
                    date_and_time.minute,
                    str(e),
                )
            ) from None


# def get_devices_from_yaml(device_data_yaml_path: str) -> Dict[str, Device]:
#     """
#     Returns a mapping from device name to device based on the data within the YAML file.

#     The yaml file path is passed in. The data from this is then read, and a series of
#     devices, each containing a probabilistic-use profile, are generated. This
#     information is returned as a mapping between the device name (or "tag") and the
#     instance of the :class:`Device` representing that device.

#     :param load_data_path:
#         The path to the load data YAML file.

#     :return:
#         A mapping between device name and :class:`Device` instance.

#     """

#     # * Call out to the utility module to read in the YAML data.

#     # ? Check that all required fields are present

#     # * Use list comprehension to generate the mapping based on the YAML data.

#     # * Return this.
