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

from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple

from .__utils__ import (
    InternalError,
    InvalidDataError,
    MissingDataError,
    ResolutionMismatchError,
    ProgrammerJudgementFault,
    BaseDailyProfile,
    read_yaml,
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


class _Day(enum.Enum):
    """
    Used to label the type of day, eg, weekday, Saturday, or Sunday.

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
class _MonthAndDay:
    """
    Used to label the type of day, eg, weekday, Saturday, or Sunday, and the Month.

    .. attribute:: month
        The month, stored as an integer. If zero, then the data is "monthless."

    .. attribute:: day
        The day, stored as a :class:`_Day`.

    """

    month: int
    day: _Day

    @classmethod
    def from_date(cls, date: datetime.date) -> Any:
        """
        Returns a :class:`_MonthAndDay` instance from the date.

        :param date:
            The current date, used to instantiate the class.

        """

        return cls(date.month, _Day.from_day_number(date.weekday()))

    def __hash__(self) -> int:
        """
        Returns a hash so the class can be keyed.

        :return:
            A value representing a unique hash of the class.

        """

        return hash(self.month * 4 + self.day.value)


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
        self, resolution=1, profile: Dict[datetime.time, float] = None
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
) -> Tuple[ProfileType, Dict[_MonthAndDay, LoadProfile]]:
    """
    Processs a CSV data file and return a mapping of month and day to profile.

    :param csv_file_path:
        The path to the csv data file.

    :return:
        A `tuple` containing the profile type and a  mapping between the
        :class:`_MonthAndDay` and the load profile as a dict.

    """

    # @@@ For now, the fact that the CSV data is an electricity profile is hard-
    # @@@ -coded in. Ideally, there would be some regex matching here.
    profile_type = ProfileType.ELECTRICITY

    electrical_load_profile: Dict[_MonthAndDay, LoadProfile] = collections.defaultdict(
        LoadProfile
    )

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
            electrical_load_profile[_MonthAndDay(1, _Day(1))].update(
                {time: float(row[2])}
            )
            electrical_load_profile[_MonthAndDay(1, _Day(2))].update(
                {time: float(row[6])}
            )
            electrical_load_profile[_MonthAndDay(2, _Day(1))].update(
                {time: float(row[10])}
            )
            electrical_load_profile[_MonthAndDay(2, _Day(2))].update(
                {time: float(row[14])}
            )
            electrical_load_profile[_MonthAndDay(3, _Day(1))].update(
                {time: float(row[18])}
            )
            electrical_load_profile[_MonthAndDay(3, _Day(2))].update(
                {time: float(row[22])}
            )
            electrical_load_profile[_MonthAndDay(4, _Day(1))].update(
                {time: float(row[26])}
            )
            electrical_load_profile[_MonthAndDay(4, _Day(2))].update(
                {time: float(row[30])}
            )
            electrical_load_profile[_MonthAndDay(5, _Day(1))].update(
                {time: float(row[34])}
            )
            electrical_load_profile[_MonthAndDay(5, _Day(2))].update(
                {time: float(row[38])}
            )
            electrical_load_profile[_MonthAndDay(6, _Day(1))].update(
                {time: float(row[42])}
            )
            electrical_load_profile[_MonthAndDay(6, _Day(2))].update(
                {time: float(row[46])}
            )
            electrical_load_profile[_MonthAndDay(7, _Day(1))].update(
                {time: float(row[50])}
            )
            electrical_load_profile[_MonthAndDay(7, _Day(2))].update(
                {time: float(row[54])}
            )
            electrical_load_profile[_MonthAndDay(8, _Day(1))].update(
                {time: float(row[58])}
            )
            electrical_load_profile[_MonthAndDay(8, _Day(2))].update(
                {time: float(row[62])}
            )
            electrical_load_profile[_MonthAndDay(9, _Day(1))].update(
                {time: float(row[66])}
            )
            electrical_load_profile[_MonthAndDay(9, _Day(2))].update(
                {time: float(row[70])}
            )
            electrical_load_profile[_MonthAndDay(10, _Day(1))].update(
                {time: float(row[74])}
            )
            electrical_load_profile[_MonthAndDay(10, _Day(2))].update(
                {time: float(row[78])}
            )
            electrical_load_profile[_MonthAndDay(11, _Day(1))].update(
                {time: float(row[82])}
            )
            electrical_load_profile[_MonthAndDay(11, _Day(2))].update(
                {time: float(row[86])}
            )
            electrical_load_profile[_MonthAndDay(12, _Day(1))].update(
                {time: float(row[90])}
            )
            electrical_load_profile[_MonthAndDay(12, _Day(2))].update(
                {time: float(row[94])}
            )

    return profile_type, electrical_load_profile


def _process_yaml(
    yaml_file_path: str,
) -> Tuple[ProfileType, Dict[_MonthAndDay, LoadProfile]]:
    """
    Process a YAML file and return a tuple containing the profile type and profile map.

    :param yaml_file_path:
        The path to the yaml data file.

    :return:
        A `tuple` containing the profile type, stored as a :class:`ProfileType`, and a
        mapping between the month and day and the profile.

    """

    yaml_data = read_yaml(yaml_file_path)

    # @@@ For now, the YAML data only concerns the hot-water demand.
    profile_type = ProfileType.HOT_WATER
    load_profile = LoadProfile(
        3600,
        {
            datetime.datetime.strptime(key, "%H:%M"): float(value)
            for key, value in yaml_data["hot_water"]["seasonless"]["dayless"].items()
        },
    )
    return profile_type, {_MonthAndDay(0, _Day(0)): load_profile}


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
        seasonal_load_profiles: Dict[
            ProfileType, Dict[_MonthAndDay, Dict[_Day, LoadProfile]]
        ],
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
        month_and_day = _MonthAndDay.from_date(date_and_time.date())

        try:
            return self._seasonal_load_profiles[profile_type][month_and_day][
                date_and_time.time()
            ]
        except (KeyError, ValueError):
            pass

        # * Attempt to get dayless data.
        month_and_day.day = _Day(0)
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
    def from_data(cls, data_file_paths: Set[str]) -> Any:
        """
        Returns a :class:`LoadSystem` based on the paths to various data files.

        :param data_file_paths:
            A set of paths to various data files.

        :return:
            A :class:`LoadSystem` instance, instantiated from the passed in data files.

        """

        seasonal_load_profiles: Dict[ProfileType, Dict[_MonthAndDay, LoadProfile]] = {
            ProfileType.ELECTRICITY: collections.defaultdict(LoadProfile),
            ProfileType.HOT_WATER: collections.defaultdict(LoadProfile),
        }

        for data_file_name in data_file_paths:
            if data_file_name.endswith(".csv"):
                profile_type, profile_data = _process_csv(data_file_name)
            elif data_file_name.endswith(".yaml"):
                profile_type, profile_data = _process_yaml(data_file_name)
            else:
                raise ProgrammerJudgementFault(
                    "Only .csv and .yaml files supported for load profiles."
                )
            seasonal_load_profiles[profile_type].update(profile_data)

        return cls(seasonal_load_profiles)