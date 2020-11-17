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

import datetime

from typing import Any, Dict

import yaml

__all__ = (
    "Device",
    "get_devices_from_yaml",
)


class Device:
    """
    Represents a single device that can be conected, either electrical or thermal.

    .. attribute:: name
        A `str` giving a label with which the device can be referenced.

    """

    # Private Attributes:
    #
    # .. attribute:: _load
    #   The device load, when used, measured in Watts.
    #
    # .. attribute:: _weekday_daytime_usage_probability
    #   The probability that the device is used during the daytime during the week.
    #
    # .. attribute:: _weekend_daytime_usage_probability
    #   The probability that the device is used during the daytime over the weekend.
    #
    # .. attribute:: _weekday_nighttime_usage_probability
    #   The probability that the device is used during the night during the week.
    #
    # .. attribute:: _weekend_nighttime_usage_probability
    #   The probability that the device is used during the night over the weekend.
    #

    def __init__(
        self,
        name: str,
        load: float,
        weekday_daytime_usage_probability: float,
        weekend_daytime_usage_probability: float,
        weekday_nighttime_usage_probability: float,
        weekend_nighttime_usage_probability: float,
    ) -> None:
        """
        Instantiate a :class:`Device` instance.

        :param name:
            A label to assign as the name of the device.

        :param load:
            The load of the device, when used, measured in Watts.

        :param weekday_daytime_usage_probability:
            The probability that the device is used during the daytime during the week.

        :param weekend_daytime_usage_probability:
            The probability that the device is used during the daytime over the weekend.

        :param weekday_nighttime_usage_probability:
            The probability that the device is used during the night during the week.

        :param weekend_nighttime_usage_probability:
            The probability that the device is used during the night over the weekend.

        """

        self.name = name
        self._load = load
        self._weekday_daytime_usage_probability = weekday_daytime_usage_probability
        self._weekend_daytime_usage_probability = weekend_daytime_usage_probability
        self._weekday_nighttime_usage_probability = weekday_nighttime_usage_probability
        self._weekend_nighttime_usage_probability = weekend_nighttime_usage_probability

    @classmethod
    def from_dict(cls, dict_entry: Dict[Any, Any]) -> Any:
        """
        Returns a :class:`Device` instance from yaml data in the form of a dictionary.

        :param dict_entry:
            A `dict` representing the device based on raw yaml data.

        :return:
            A :class:`Device` instance based on the data.

        """

        # * Check that all the required fields are present
        # ? Maybe this checking should be done when the YAML data is read in?

        # * Generate, and return, a Device instance as appropriate.

        return cls("", 0, 0, 0, 0, 0)

    def load(self, date_and_time: datetime.datetime) -> float:
        """
        Returns the current load of the device, in Watts, based on the date and time.

        :param date_and_date:
            The current date and time.

        :return:
            The load of the device in Watts based on the probability of it being used.

        """

        # * Determine whether it is a weekend or weekday and whether it's day or night.

        # * Determine the instantaneous load of the device and return it.


def get_devices_from_yaml(load_data_path: str) -> Dict[str, Device]:
    """
    Returns a mapping from device name to device based on the data within the YAML file.

    The yaml file path is passed in. The data from this is then read, and a series of
    devices, each containing a probabilistic-use profile, are generated. This
    information is returned as a mapping between the device name (or "tag") and the
    instance of the :class:`Device` representing that device.

    :param load_data_path:
        The path to the load data YAML file.

    :return:
        A mapping between device name and :class:`Device` instance.

    """

    # * Call out to the utility module to read in the YAML data.

    # ? Check that all required fields are present

    # * Use list comprehension to generate the mapping based on the YAML data.

    # * Return this.
