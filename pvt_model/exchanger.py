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

    """

    # Private Attributes:
    #
    # .. attribute:: _efficiency
    #   The efficiency of the heat exchanger, defined between 0 and 1.

    def __init__(self, efficiency) -> None:
        """
        Instantiate a heat exchanger instance.

        :param efficiency:
            The efficiency of the heat exchanger, defined between 0 and 1.

        """

        self._efficiency = efficiency

    def update(
        self,
        water_tank: tank.Tank,
        input_water_temperature: float,
        input_water_flow_rate: float,
        input_water_heat_capacity: float,
        ambient_tank_temperature: float,
        mains_water_temperature: float,
        demand_water_flow_rate: float,
    ) -> float:
        """
        Updates the tank temperature based on the input water temperature.

        :param tank:
            A :class:`tank.Tank` representing the hot-water tank being filled.

        :param input_water_temperature:
            The temperature of the water being inputted to the heat exchanger, measured
            in Kelvin.

        :param input_water_flow_rate:
            The flow rate of water entering the exchanger from the PV-T panel, measured
            in kilograms per unit time step.

        :param input_water_heat_capacity:
            The heat capacity of the water used to feed the heat exchanger, measured in
            Joules per kilogram Kelvin.

        :param ambient_tank_temperature:
            The temperature of the air surrounding the tank, measured in Kelvin.

        :param mains_water_temperature:
            The temperature of the water used to feed the system, usually that of the
            mains water supply, measured in Kelvin.

        :param demand_water_flow_rate:
            The flow rate of water required by the end user, measured in cubic meters
            per time step.
            ??? Again, the time interval here may be SI seconds or model time step.
            ??? This needs thinking about... :p

        :return:
            The output water temperature from the heat exchanger, measured in Kelvin.

        """

        heat_lost = (
            water_tank.area
            * water_tank.heat_loss_coefficient
            * (water_tank.temperature - ambient_tank_temperature)
        )

        heat_delivered = (
            demand_water_flow_rate
            * water_tank.heat_capacity
            * (water_tank.temperature - mains_water_temperature)
        )

        # If the water inputted to the exchanger is less than the tank temperature, then
        # run it straight back into the next cycle.
        if input_water_temperature <= water_tank.temperature:
            water_tank.temperature = water_tank.temperature - (
                heat_lost + heat_delivered
            ) / (water_tank.mass * water_tank.heat_capacity)
            return input_water_temperature

        # Apply the first law of Thermodynamics to determine the output water
        # temperature from the heat exchanger.
        output_water_temperature = input_water_temperature - self._efficiency * (
            input_water_temperature - water_tank.temperature
        )

        # Determine the new tank temperature using properties of the tank.
        heat_added = (
            self._efficiency * input_water_flow_rate * input_water_heat_capacity
        ) * (input_water_temperature - water_tank.temperature)

        water_tank.temperature = (heat_added - heat_lost - heat_delivered) / (
            water_tank.mass * water_tank.heat_capacity
        ) + water_tank.temperature

        # Return the output temperature of the heat exchanger.
        return output_water_temperature
