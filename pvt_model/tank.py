#!/usr/bin/python3.7
########################################################################################
# tankg.py - The tankg module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The tank module for the PV-T model.

This module represents the hot-water tank.

"""

__all__ = ("Tank",)


class Tank:
    """
    Represents a hot-water tank.

    .. attribute:: temperature
        The temperature of the hot-water tank, measured in Kelvin.

    .. attribute:: mass
        The mass of water in the hot-water tank, measured in kilograms.

    .. attribute:: heat_capacity
        The heat capacity of the water, measured in Joules per kilogram Kelvin.

    .. attribute:: area
        The surface area of the tank, measured in meters squared.

    .. attribute:: heat_loss_coefficient
        The heat lost from the tank, measured in Joules per meter squared Kelvin.

    """

    def __init__(
        self,
        temperature: float,
        mass: float,
        heat_capacity: float,
        area: float,
        heat_loss_coefficient: float,
    ) -> None:
        """
        Instantiate a hot-water tank.

        :param temperature:
            The temperature of the water within the tank when initilialsed, measured in
            Kelvin.

        :param mass:
            The mass of water that can be held within the tank, measured in kilograms.

        :param heat_capacity:
            The heat capacity of water within the tank, measured in Joules per kilogram
            Kelvin.

        :param area:
            The surface area of the tank, measured in meters squared.

        :param heat_loss_coefficient:
            The heat lost from the tank, measured in Joules per meter squared Kelvin.

        """

        self.temperature = temperature
        self.mass = mass
        self.heat_capacity = heat_capacity
        self.area = area
        self.heat_loss_coefficient = heat_loss_coefficient
