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

from typing import Tuple

from . import tank

__all__ = ("Exchanger",)


class Exchanger:
    """
    Represents a physical heat exchanger within a hot-water tank.

    """

    # Private Attributes:
    #
    # .. attribute:: _efficiency
    #   The efficiency of the heat exchanger, defined between 0 and 1.
    #

    def __init__(self, efficiency) -> None:
        """
        Instantiate a heat exchanger instance.

        :param efficiency:
            The efficiency of the heat exchanger, defined between 0 and 1.

        """

        self._efficiency = efficiency

    def __repr__(self) -> str:
        """
        Returns a nice representation of the heat exchanger.

        :return:
            A `str` giving a nice representation of the heat exchanger.

        """

        return f"Exchanger(efficiency: {self._efficiency})"

    def update(
        self,
        input_water_heat_capacity: float,
        input_water_mass: float,
        input_water_temperature: float,
        water_tank: tank.Tank,
    ) -> Tuple[float, float]:
        """
        Updates the tank temperature based on the input water temperature.

        :param input_water_heat_capacity:
            The heat capacity of the water used to feed the heat exchanger, measured in
            Joules per kilogram Kelvin.

        :param input_water_mass:
            The flow rate of water entering the exchanger from the PV-T panel, measured
            in kilograms per unit time step. As this has been multiplied by the number
            of seconds per unit time step, it is effectively just the mass that has
            passed through the exchanger and delivered some heat.

        :param input_water_temperature:
            The temperature of the water being inputted to the heat exchanger, measured
            in Kelvin.

        :param tank:
            A :class:`tank.Tank` representing the hot-water tank being filled.

        :return:
            The output water temperature from the heat exchanger, measured in Kelvin,
            and the heat added to the hot-water tank, measured in Joules, as a Tuple.

        """

        # If the water inputted to the exchanger is less than the tank temperature, then
        # run it straight back into the next cycle.
        if input_water_temperature <= water_tank.temperature:
            return input_water_temperature, 0

        # Determine the new tank temperature using properties of the tank.
        # Determine the heat added in Joules. Because the input water flow rate is
        # measured in kilograms per time step, this can be used as is as a total mass
        # flow param in kilograms.
        heat_added = (
            # self._efficiency
            # * input_water_mass  # [kg]
            # * input_water_heat_capacity  # [J/kg*K]
            57300  # [W/K]
        ) * (
            input_water_temperature - water_tank.temperature
        )  # [K]

        # @@@
        # >>> Potential incorrect equation.
        # Apply the first law of Thermodynamics to determine the output water
        # temperature from the heat exchanger.
        output_water_temperature = input_water_temperature - self._efficiency * (
            input_water_temperature - water_tank.temperature
        )
        # <<< End of potential incorrect equation.

        # Return the output temperature of the heat exchanger.
        return (
            output_water_temperature,
            heat_added,
        )
