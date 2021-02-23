#!/usr/bin/python3.7
########################################################################################
# tank.py - The tank module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The tank module for the PV-T model.

This module represents the hot-water tank.

"""

from .constants import HEAT_CAPACITY_OF_WATER

__all__ = (
    "net_enthalpy_gain",
    "Tank",
)


def net_enthalpy_gain(
    delivery_temperature: float,
    mains_water_temperature: float,
    water_demand_volume: float,
) -> float:
    """
    Computes the net enthalpy gain by the tank due to delivering water, in Watts.

    :param delivery_temperature:
        The delivery temperature of the hot-water tank, measured in Kelvin.

    :param mains_water_temperature:
        The temperature of the mains water used to replace any water removed from the
        hot-water tank, measured in Kelvin.

    :param water_demand_volume:
        The volume of water removed from the tank, measured in litres. This will be the
        same as the mass of water removed from the tank, measured in kilograms, due to
        the density of water.

    """

    return (
        water_demand_volume  # [kg]
        * HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        * (mains_water_temperature - delivery_temperature)  # [K]
    )


class Tank:
    """
    Represents a hot-water tank.

    .. attribute:: area
        The surface area of the tank, measured in meters squared.

    .. attribute:: diameter
        The diameter of the tank, measured in meters.

    .. attribute:: heat_capacity
        The heat capacity of the water, measured in Joules per kilogram Kelvin.

    .. attribute:: heat_loss_coefficient
        The heat lost from the tank, measured in Watts per meter squared Kelvin.

    .. attribute:: mass
        The mass of water in the hot-water tank, measured in kilograms.

    """

    def __init__(
        self,
        area: float,
        diameter: float,
        heat_capacity: float,
        heat_loss_coefficient: float,
        mass: float,
    ) -> None:
        """
        Instantiate a hot-water tank.

        :param area:
            The surface area of the tank, measured in meters squared.

        :param diameter:
            The diameter of the hot-water tank being mnodelled, measured in meters.

        :param heat_capacity:
            The heat capacity of water within the tank, measured in Joules per kilogram
            Kelvin.

        :param heat_loss_coefficient:
            The heat lost from the tank, measured in Watts per meter squared Kelvin.

        :param mass:
            The mass of water that can be held within the tank, measured in kilograms.

        """

        self.area = area
        self.diameter = diameter
        self.heat_capacity = heat_capacity
        self.heat_loss_coefficient = heat_loss_coefficient
        self.mass = mass

    def __repr__(self) -> str:
        """
        Returns a nice representation of the hot-water tank.

        :return:
            A `str` giving a nice representation of the hot-water tank.

        """

        return (
            "Tank("
            f"area: {self.area}m^2, "
            f"diameter: {self.diameter}m, "
            f"heat_capacity: {self.heat_capacity}J/kg*K, "
            f"heat_loss_coefficient: {self.heat_loss_coefficient}W/m^2*K, "
            f"mass: {self.mass}kg, "
            ")"
        )

    def heat_loss(
        self, ambient_tank_temperature: float, tank_temperature: float
    ) -> float:
        """
        Computes the heat loss through the walls of the tank, measured in Watts.

        A value of 573 W/K is used by Maria's paper.

        :param ambient_temperature:
            The temperature of the air surrounding the hot-water tank, measured in
            Kelvin.

        :param tank_temperature:
            The temperature of the fluid within the tank, measured in Kelvin.

        :return:
            The heat loss from the tank, measured in Watts.

        """

        return (
            self.area  # [m^2]
            * self.heat_loss_coefficient  # [W/m^2*K]
            * (tank_temperature - ambient_tank_temperature)  # [K]
        )  # [W]
