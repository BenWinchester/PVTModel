#!/usr/bin/python3.7
########################################################################################
# exchanger.py - The exchanger module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The exchanger module for the PV-T model.

This module represents the heat exchanger within the hot-water tank.

"""

from . import tank

__all__ = ("Exchanger",)


class Exchanger:
    """
    Represents a physical heat exchanger within a hot-water tank.

    .. attribute::

    """

    def __init__(self) -> None:
        """
        Instantiate a heat exchanger instance.

        """

    def update(self, tank: tank.Tank, input_water_temperature: float) -> float:
        """
        Updates the tank temperature based on the input water temperature.

        :param tank:
            A :class:`tank.Tank` representing the hot-water tank being filled.

        :param input_water_temperature:
            The temperature of the water being inputted to the heat exchanger, measured
            in Kelvin.

        :return:
            The output water temperature from the heat exchanger, measured in Kelvin.

        """

        # * Determine the new tank temperature using properties of the tank.

        # * Apply the first law of Thermodynamics to determine the output water
        # * temperature from the heat exchanger.

        # * Return this output temperature.

        return 0
