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
import csv
import datetime
import enum

import json

from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple

from .__utils__ import (
    MissingDataError,
    ProgrammerJudgementFault,
    BaseDailyProfile,
)

__all__ = (
    "LoadData",
    "LoadProfile",
    "LoadSystem",
    "ProfileType",
)


################
# Data Classes #
################


class ProfileType(enum.Enum):
    """
    Used to label whether a profile is a hot-water or electrical (or other) profile.

    """

    ELECTRICITY = 0
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


class _DayType(enum.Enum):
    """
    Contains information about the type of day.

    .. attribute:: WEEKDAY
        A weekday is being represented here.

    .. attribute:: WEEKEND
        A weekend day is being represented here.

    .. attribute:: DAYLESS
        The type of day, i.e., weekday or weekend, is irrelevant. This information is
        carried around the code as a :class:`_DayType`.DAYLESS instance.

    """

    DAYLESS = 0
    WEEKDAY = 1
    WEEKEND = 2

    @classmethod
    def from_day_number(cls, day_number: int) -> Any:
        """
        Return a :class:`_Day` instance based on the day of the week being processed.

        :param day_number:
            The day number, ie mon = 1, tue = 2, etc..

        :return:
            A :class:`_Day` instance based on this.

        """

        if not isinstance(day_number, int):
            raise Exception("The day must be an integer!")

        if 0 <= day_number <= 4:
            return cls(1)
        if 5 <= day_number <= 6:
            return cls(2)
        if day_number == 7:
            return cls(0)
        raise Exception("The day must fall between 0 and 6 inclusive, or 7 for no day.")


@dataclass
class _MonthAndDayType:
    """
    Contains information about the month and the weekday.

    .. attribute:: month
        The month, expressed as an integer.

    .. attribute:: day_type
        The day type, expressed as a :class:`_DayType`.

    """

    month: int
    day_type: _DayType

    def __hash__(self) -> int:
        """
        Generates a hash s.t. :class:`_MonthAndDayType` instances can be used as keys.

        :return:
            A hash of the :class:`_MonthAndDayType` instnace.

        """

        return hash(self.month * 31 + self.day_type.value)

    @classmethod
    def from_date(cls, date: datetime.date) -> Any:
        """
        Returns a :class:`_MonthAndWeekday` instance based on the date.

        :param date:
            The date for which to generate a month and weekday instance, passed in as a
            :class:`datetime.date` instance.

        :return:
            An instantiated :class:`_MonthAndWeekday` instance.

        """

        return cls(date.month, _DayType.from_day_number(date.weekday()))


class LoadProfile(BaseDailyProfile):
    """
    Represents a load profile.

    .. attribute:: resolution
        The time-step resolution of the profile, measured in seconds.

    """

    # Private Attributes:
    #
    #  .. attribute:: _profile
    #   A mapping of hour to the load at that hour.
    #

    def __init__(
        self,
        profile: Dict[datetime.time, float] = None,
        resolution=1,
    ) -> None:
        """
        Instantiate a load profile.

        :param resolution:
            The resolution of the profile, measured in seconds.

        :param profile:
            The profile data: a mapping of time to the load at that time step.

        """

        super().__init__(profile)

        self.resolution = resolution


#############################
# Internal Helper Functions #
#############################


def _process_csv(
    csv_file_path: str,
) -> Tuple[ProfileType, Dict[_MonthAndDayType, LoadProfile]]:
    """
    Processs a CSV data file and return a mapping of month and day to profile.

    :param csv_file_path:
        The path to the csv data file.

    :return:
        A `tuple` containing the profile type and a  mapping between the
        :class:`_MonthAndDayType` and the load profile as a dict.

    """

    # @@@ For now, the fact that the CSV data is an electricity profile is hard-
    # @@@ -coded in. Ideally, there would be some regex matching here.
    profile_type = ProfileType.ELECTRICITY

    electrical_load_profile: Dict[
        _MonthAndDayType, LoadProfile
    ] = collections.defaultdict(LoadProfile)

    # Open the csv file, and cycle through the rows, processing them.
    with open(csv_file_path, "r") as csv_file:
        file_reader = csv.reader(csv_file, delimiter=",")

        # Cycle through the rows, and process them accordingly.
        for index, row in enumerate(file_reader):
            # Skip the title rows
            if index < 2:
                continue

            # Determine the current time.
            time = datetime.datetime.strptime(row[0], "%H:%M").time()

            # Set the various profile values accordingly.
            electrical_load_profile[_MonthAndDayType(1, _DayType(1))].update(
                {time: float(row[2])}
            )
            electrical_load_profile[_MonthAndDayType(1, _DayType(2))].update(
                {time: float(row[6])}
            )
            electrical_load_profile[_MonthAndDayType(2, _DayType(1))].update(
                {time: float(row[10])}
            )
            electrical_load_profile[_MonthAndDayType(2, _DayType(2))].update(
                {time: float(row[14])}
            )
            electrical_load_profile[_MonthAndDayType(3, _DayType(1))].update(
                {time: float(row[18])}
            )
            electrical_load_profile[_MonthAndDayType(3, _DayType(2))].update(
                {time: float(row[22])}
            )
            electrical_load_profile[_MonthAndDayType(4, _DayType(1))].update(
                {time: float(row[26])}
            )
            electrical_load_profile[_MonthAndDayType(4, _DayType(2))].update(
                {time: float(row[30])}
            )
            electrical_load_profile[_MonthAndDayType(5, _DayType(1))].update(
                {time: float(row[34])}
            )
            electrical_load_profile[_MonthAndDayType(5, _DayType(2))].update(
                {time: float(row[38])}
            )
            electrical_load_profile[_MonthAndDayType(6, _DayType(1))].update(
                {time: float(row[42])}
            )
            electrical_load_profile[_MonthAndDayType(6, _DayType(2))].update(
                {time: float(row[46])}
            )
            electrical_load_profile[_MonthAndDayType(7, _DayType(1))].update(
                {time: float(row[50])}
            )
            electrical_load_profile[_MonthAndDayType(7, _DayType(2))].update(
                {time: float(row[54])}
            )
            electrical_load_profile[_MonthAndDayType(8, _DayType(1))].update(
                {time: float(row[58])}
            )
            electrical_load_profile[_MonthAndDayType(8, _DayType(2))].update(
                {time: float(row[62])}
            )
            electrical_load_profile[_MonthAndDayType(9, _DayType(1))].update(
                {time: float(row[66])}
            )
            electrical_load_profile[_MonthAndDayType(9, _DayType(2))].update(
                {time: float(row[70])}
            )
            electrical_load_profile[_MonthAndDayType(10, _DayType(1))].update(
                {time: float(row[74])}
            )
            electrical_load_profile[_MonthAndDayType(10, _DayType(2))].update(
                {time: float(row[78])}
            )
            electrical_load_profile[_MonthAndDayType(11, _DayType(1))].update(
                {time: float(row[82])}
            )
            electrical_load_profile[_MonthAndDayType(11, _DayType(2))].update(
                {time: float(row[86])}
            )
            electrical_load_profile[_MonthAndDayType(12, _DayType(1))].update(
                {time: float(row[90])}
            )
            electrical_load_profile[_MonthAndDayType(12, _DayType(2))].update(
                {time: float(row[94])}
            )

    return profile_type, electrical_load_profile


def _process_json(
    json_file_path: str,
) -> Tuple[ProfileType, Dict[_MonthAndDayType, LoadProfile]]:
    """
    Processs a JSON data file and return a mapping of month and day to profile.

    :param json_file_path:
        The path to the json data file.

    :return:
        A `tuple` containing the profile type and a  mapping between the
        :class:`_MonthAndDayType` and the load profile as a dict.

    """

    if "thermal" in json_file_path:
        profile_type = ProfileType.HOT_WATER
    elif "electrical" in json_file_path:
        profile_type = ProfileType.ELECTRICITY
    else:
        raise ProgrammerJudgementFault(
            "The profle type supplied is not electrical or thermal. See load module."
        )

    # @@@ For now, the data is stored irrespective of month and day. This is in order to
    # @@@ match as closely as possible to Maria's data set for July.
    load_profile: Dict[_MonthAndDayType, LoadProfile] = collections.defaultdict(
        LoadProfile
    )

    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    # Process the data.
    try:
        load_profile[_MonthAndDayType(0, _DayType(0))] = LoadProfile(
            {
                datetime.datetime.strptime(key, "%H:%M:%S.%f").time(): float(value)
                for key, value in json_data.items()
            }
        )
        return profile_type, load_profile
    except ValueError:
        pass

    # Attempt using a different time format.
    try:
        load_profile[_MonthAndDayType(0, _DayType(0))] = LoadProfile(
            {
                datetime.datetime.strptime(key, "%H:%M:%S").time(): float(value)
                for key, value in json_data.items()
            }
        )
        return profile_type, load_profile
    except ValueError:
        pass

    # Attempt using a different time format.
    try:
        load_profile[_MonthAndDayType(0, _DayType(0))] = LoadProfile(
            {
                datetime.datetime.strptime(key, "%H:%M").time(): float(value)
                for key, value in json_data.items()
            }
        )
        return profile_type, load_profile
    except ValueError as e:
        raise ProgrammerJudgementFault(
            f"The load profile data is of an incompatible time format: {str(e)}"
        ) from None


#########################
# External Load Classes #
#########################


class LoadSystem:
    """
    Represents a system, eg, a household, potential of generating loads.

    """

    # Private Attributes:
    #
    # .. attribute:: _monthly_load_profiles
    #   A mapping between the profile type, month, and day of the profile.
    #

    def __init__(
        self,
        seasonal_load_profiles: Dict[ProfileType, Dict[_MonthAndDayType, LoadProfile]],
    ) -> None:
        """
        Instantiate a Load System.

        :param seasonal_load_profiles:
            A mapping representing the internal structure of how load profiles are to be
            stored.

        """

        self._seasonal_load_profiles = seasonal_load_profiles

    def __getitem__(self, index: Tuple[ProfileType, datetime.datetime]) -> float:
        """
        Returns the load value based on the profile type wanted and the date and time.

        :param index:
            A `tuple` containing the profile type wanted and the date and time for which
            to fetch the value.

        :return:
            The value of the given profile at the given date and time.

        """

        profile_type, date_and_time = index
        month_and_day = _MonthAndDayType.from_date(date_and_time.date())

        try:
            return self._seasonal_load_profiles[profile_type][month_and_day][
                date_and_time.time()
            ]
        except (KeyError, ValueError):
            pass

        # * Attempt to get dayless data.
        month_and_day.day_type = _DayType(0)
        try:
            return self._seasonal_load_profiles[profile_type][month_and_day][
                date_and_time.time()
            ]
        except (KeyError, ValueError):
            pass

        # * Attempt to get seasonless data
        month_and_day.month = 0
        try:
            return self._seasonal_load_profiles[profile_type][month_and_day][
                date_and_time.time()
            ]
        except (KeyError, ValueError) as e:
            raise MissingDataError(
                "Could not find profile data for given key. Key: {}. Error: {}".format(
                    str(index),
                    str(e),
                )
            ) from None

    @classmethod
    def from_data(cls, data_file_paths: Set[str]) -> Any:  # type: ignore
        """
        Returns a :class:`LoadSystem` based on the paths to various data files.

        :param data_file_paths:
            A set of paths to various data files.

        :return:
            A :class:`LoadSystem` instance, instantiated from the passed in data files.

        """

        seasonal_load_profiles: Dict[
            ProfileType, Dict[_MonthAndDayType, LoadProfile]
        ] = {
            ProfileType.ELECTRICITY: collections.defaultdict(LoadProfile),
            ProfileType.HOT_WATER: collections.defaultdict(LoadProfile),
        }

        for data_file_name in data_file_paths:
            if data_file_name.endswith(".csv"):
                profile_type, profile_data = _process_csv(data_file_name)
            elif data_file_name.endswith(".json"):
                profile_type, profile_data = _process_json(data_file_name)
            else:
                raise ProgrammerJudgementFault(
                    "Only .csv and .json files supported for load profiles. "
                    f"Filename attempted: {data_file_name}"
                )
            seasonal_load_profiles[profile_type].update(profile_data)

        return cls(seasonal_load_profiles)
