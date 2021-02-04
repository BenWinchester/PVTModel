#!/usr/bin/python3.7
########################################################################################
# pvt_panel/glass.py - Represents a glass within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The glass module for the PV-T model.

This module represents a glass layer within a PV-T panel.

"""

from ..__utils__ import WeatherConditions
from .__utils__ import OpticalLayer, radiative_heat_transfer, wind_heat_transfer

__all__ = ("Glass",)


class Glass(OpticalLayer):
    """
    Represents the glass (upper) layer of the PV-T panel.

    """

    diffuse_reflection_coefficient = 0.16

    def update(
        self,
        heat_input: float,
        weather_conditions: WeatherConditions,
    ) -> float:
        """
        Update the internal properties of the PV layer based on external factors.

        :param heat_input:
            The heat inputted to the glass layer, measured in Watts.

        :param weather_conditions:
            The weather conditions at the current time step.

        :return:
            The heat lost upwards from the glass layer, measured in Joules.

        """

        upward_heat_losses = wind_heat_transfer(
            contact_area=self.area,
            destination_temperature=weather_conditions.ambient_temperature,
            source_temperature=self.temperature,
            wind_heat_transfer_coefficient=weather_conditions.wind_heat_transfer_coefficient,
        ) + radiative_heat_transfer(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            radiative_contact_area=self.area,
            source_emissivity=self.emissivity,
            source_temperature=self.temperature,
        )  # [W]

        # This heat input, in Watts, is supplied throughout the duration, and so does
        # not need to be multiplied by the resolution.
        self.temperature = self.temperature + (  # [K]
            heat_input - upward_heat_losses
        ) / (  # [W]
            self._mass * self._heat_capacity
        )  # [kg] * [J/kg*K]

        return upward_heat_losses  # [W]
