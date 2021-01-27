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
from .__utils__ import OpticalLayer

__all__ = ("Glass",)


class Glass(OpticalLayer):
    """
    Represents the glass (upper) layer of the PV-T panel.

    """

    def update(
        self,
        heat_input: float,
        internal_resolution: float,
        weather_conditions: WeatherConditions,
    ) -> None:
        """
        Update the internal properties of the PV layer based on external factors.

        :param heat_input:
            The heat inputted to the glass layer, measured in Watts.

        :param internal_resolution:
            The resolution of the simulation currently being run, measured in seconds.

        :param weather_conditions:
            The weather conditions at the current time step.

        """

        upward_heat_losses = self._layer_to_air_convective_transfer(
            weather_conditions.ambient_temperature,
            fraction_emitting=1,
            wind_heat_transfer_coefficient=weather_conditions.wind_heat_transfer_coefficient,
        ) + self._layer_to_sky_radiative_transfer(
            fraction_emitting=1, sky_temperature=weather_conditions.sky_temperature
        )  # [W]

        # This heat input, in Watts, is supplied throughout the duration, and so does
        # not need to be multiplied by the resolution.
        self.temperature = self.temperature + (  # [K]
            heat_input - upward_heat_losses
        ) * internal_resolution / (  # [W] * [seconds]
            self._mass * self._heat_capacity
        )  # [kg] * [J/kg*K]
