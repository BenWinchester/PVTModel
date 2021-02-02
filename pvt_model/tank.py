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

from .__utils__ import HEAT_CAPACITY_OF_WATER

__all__ = ("Tank",)


class Tank:
    """
    Represents a hot-water tank.

    .. attribute:: area
        The surface area of the tank, measured in meters squared.

    .. attribute:: heat_capacity
        The heat capacity of the water, measured in Joules per kilogram Kelvin.

    .. attribute:: heat_loss_coefficient
        The heat lost from the tank, measured in Watts per meter squared Kelvin.

    .. attribute:: mass
        The mass of water in the hot-water tank, measured in kilograms.

    .. attribute:: temperature
        The temperature of the hot-water tank, measured in Kelvin.

    """

    def __init__(
        self,
        area: float,
        heat_capacity: float,
        heat_loss_coefficient: float,
        mass: float,
        temperature: float,
    ) -> None:
        """
        Instantiate a hot-water tank.

        :param area:
            The surface area of the tank, measured in meters squared.

        :param heat_capacity:
            The heat capacity of water within the tank, measured in Joules per kilogram
            Kelvin.

        :param heat_loss_coefficient:
            The heat lost from the tank, measured in Watts per meter squared Kelvin.

        :param mass:
            The mass of water that can be held within the tank, measured in kilograms.

        :param temperature:
            The temperature of the water within the tank when initilialsed, measured in
            Kelvin.

        """

        self.area = area
        self.heat_capacity = heat_capacity
        self.heat_loss_coefficient = heat_loss_coefficient
        self.mass = mass
        self.temperature = temperature

    def __repr__(self) -> str:
        """
        Returns a nice representation of the hot-water tank.

        :return:
            A `str` giving a nice representation of the hot-water tank.

        """

        return (
            "Tank("
            f"area: {self.area}m^2, "
            f"heat_capacity: {self.heat_capacity}J/kg*K, "
            f"heat_loss_coefficient: {self.heat_loss_coefficient}W/m^2*K, "
            f"mass: {self.mass}kg, "
            f"temperature: {self.temperature}K"
            ")"
        )

    def update(
        self,
        heat_gain: float,
        internal_resolution: float,
        water_demand_volume: float,
        mains_water_temp: float,
        ambient_tank_temperature: float,
    ) -> float:
        """
        Updates the tank temperature when a certain volume of hot water is demanded.

        :param internal_resolution:
            The internal_resolution of the model currently being run, measured in seconds.

        :param water_demand_volume:
            The volume of hot water demanded by the end user, measured in litres.

        :param mains_water_temp:
            The temperature of the mains water used to fill the tank, measured in
            Kelvin.

        :param ambient_tank_temperature:
            The temperature of the air surrounding the tank, measured in Kelvin.

        :return:
            The temperature of the hot-water delivered.

        """

        # We need to multiply by the internal_resolution in order to compute the total heat lost
        # from the tank during the time duration.
        heat_loss = (
            self.area
            * self.heat_loss_coefficient
            * (self.temperature - ambient_tank_temperature)
        ) * internal_resolution

        delivery_temp = self.temperature

        net_enthalpy_gain = (
            water_demand_volume
            * HEAT_CAPACITY_OF_WATER
            * (mains_water_temp - delivery_temp)
        )

        # We lose this heat, as we're considering things as 30min "block" inputs and
        # outputs.
        self.temperature += (heat_gain - heat_loss + net_enthalpy_gain) / (
            self.mass * self.heat_capacity
        )

        # The new temperature is computed by a mass-weighted average of the temperatures
        # of the various water sources that go into making up the new content of the
        # hot-water tank.
        # self.temperature = (
        #     self.temperature * (self.mass - water_demand_volume)
        #     + mains_water_temp * water_demand_volume
        # ) / self.mass

        return delivery_temp
