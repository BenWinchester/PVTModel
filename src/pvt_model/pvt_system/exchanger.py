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

__all__ = ("Exchanger",)


class Exchanger:
    """
    Represents a physical heat exchanger within a hot-water tank.

    .. attribute:: efficiency
        The efficiency of the heat exchanger, defined between 0 and 1.

    """

    def __init__(self, efficiency) -> None:
        """
        Instantiate a heat exchanger instance.

        :param efficiency:
            The efficiency of the heat exchanger, defined between 0 and 1.

        """

        self.efficiency = efficiency

    def __repr__(self) -> str:
        """
        Returns a nice representation of the heat exchanger.

        :return:
            A `str` giving a nice representation of the heat exchanger.

        """

        return f"Exchanger(efficiency: {self.efficiency})"

    def get_heat_addition(
        self,
        input_water_mass_flow_rate: float,
        input_water_heat_capacity: float,
        input_water_temperature: float,
        water_tank_temperature: float,
    ) -> float:
        """
        Computes the heat added to the hot-water tank.

        :param input_water_heat_capacity:
            The heat capacity of the HTF, measured in Joules per kilogram Kelvin.

        :param input_water_mass_flow_rate:
            The mass flow rate of water through the HTF side of the heat exchanger,
            measured in kilograms per second.

        :param input_water_temperature:
            The tempertaure of the HTF being inputted on the HTF side of the heat
            exchanger.

        :param water_tank_temperature:
            The hot-water tank temperature, measured in Kelvin.

        :return:
            The heat addition to the hot-water tank from the exchanger, measured in
            Watts.

        """

        tank_heat_addition: float = (
            self.efficiency
            * input_water_mass_flow_rate  # [kg/s]
            * input_water_heat_capacity  # [J/kg*K]
        ) * (
            input_water_temperature - water_tank_temperature  # [K]
        )  # [W]

        return tank_heat_addition

    def get_output_htf_temperature(
        self,
        input_water_temperature: float,
        water_tank_temperature: float,
    ) -> float:
        """
        Computes the temperature of the HTF leaving the heat exchanger, in Kelvin.

        :param input_water_temperature:
            The temperature of the HTF entering the heat exchanger, measured in Kelvin.

        :param water_tank_temperature:
            The temperature of the hot-water tank, measured in Kelvin.

        :return:
            The temperature of the HTF leaving the heat exchanger, measured in Kelvin.

        """

        exchanger_output_temperature: float = (
            input_water_temperature
            - self.efficiency * (input_water_temperature - water_tank_temperature)
        )

        return exchanger_output_temperature
