#!/usr/bin/python3.7
########################################################################################
# pvt.py - Models a PVT panel and all contained components.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The PV-T module for the PV-T model.

This module models the PV-T panel and its components, including the glass cover, PV
panel, and thermal collector. The model works by incrimenting the parameters through at
increasing time steps, and the code here aims to emaulate this.

"""

import abc
import time

from typing import Optional

from .__utils__ import WeatherConditions

__all__ = ("PVT",)


class MissingParametersError(Exception):
    """
    Raised when not all parameters have been specified that are needed to instantiate.

    """

    def __init__(self, class_name, message) -> None:
        """
        Instantiate a missing parameters error.

        :param class_name:
            The class for which insufficient parameters were specified.

        :param message:
            An appended message to display to the user.

        """

        super().__init__(
            f"Missing parameters when initialising a '{class_name}' class: {message}."
        )


class _Layer:
    """
    Represents a layer within the PV-T panel.

    .. attribute:: temperature
        The temperature of the layer, measured in Kelvin.

    """

    # Private attributes:
    #
    # .. attribute:: _mass
    #   The mass of the layer in kilograms.
    #
    # .. attribute:: _heat_capacity
    #   The heat capacity of the layer, measured in Joules per kilogram Kelvin.
    #

    def __init__(
        self, mass: float, heat_capacity: float, temperature: float = 273
    ) -> None:
        """
        Instantiate an instance of the layer class.

        :param temperature:
            The temperature at which to initialise the layer, measured in Kelvin.

        """

        self._mass = mass
        self._heat_capacity = heat_capacity
        self.temperature = temperature


class Glass(_Layer):
    """
    Represents the glass (upper) layer of the PV-T panel.

    """

    def update(
        self, pv_temp: float, wind_speed: float, ambient_temperature: float
    ) -> None:
        """
        Update the internal properties of the PV layer based on external factors.

        :param pv_temp:
            The temperature of the adjoining PV layer, measured in Kelvin, at the
            current time step.

        :param wind_speed:
            The speed of the wind, measured in meters per second.

        :param ambient_temperature:
            The temperature of the air surrounding the panel, measured in Kelvin.

        """

        # * Set the temperature of this layer appropriately.


class Collector(_Layer):
    """
    Represents the thermal collector (lower) layer of the PV-T panel.

    """

    def update(self, pv_temp: float, input_water_temperature: float) -> None:
        """
        Update the internal properties of the PV layer based on external factors.

        :param pv_temp:
            The temperature of the adjoining PV layer, measured in Kelvin, at the
            current time step.

        :param input_water_temperature:
            The temperature of the input water flow to the collector, measured in
            Kelvin, at the current time step.

        """

        # * Set the temperature of this layer appropriately.

        # ! At this point, it may be that the output water temperature from the panel
        # ! should be computed. We will see... :)


class PV(_Layer):
    """
    Represents the photovoltaic (middle) layer of the PV-T panel.

    """

    # Private attributes:
    #
    # .. attribute:: _reference_efficiency
    #   The efficiency of the PV layer at the reference temperature. Thie value varies
    #   between 1 (corresponding to 100% efficiency), and 0 (corresponding to 0%
    #   efficiency)
    #
    # .. attribute:: _reference_temperature
    #   The referencee temperature, in Kelvin, at which the reference efficiency is
    #   defined.
    #
    # .. attribute:: _thermal_coefficient
    #   The thermal coefficient for the efficiency of the panel.
    #

    def update(
        self, solar_irradiance: float, glass_temp: float, collector_temp: float
    ) -> None:
        """
        Update the internal properties of the PV layer based on external factors.

        :param solar_irradiance:
            The solar irradiance, normal to the panel, measured in Watts.

        :param glass_temp:
            The tempearture of the adjoining glass layer, measured in Kelvin, at the
            previous time step.

        :param collector_temp:
            The temperature of the adjoining collector layer, measured in Kelvin, at the
            previous time step.

        """

        # * Determine the excess heat that has been inputted into the panel.

        # * Use this to compute the rise in temperature of the PV layer.

        # * Set the temperature of this layer appropriately.

    @property
    def efficiency(self) -> float:
        """
        Returns the percentage efficiency of the PV panel based on its temperature.

        :return:
            A decimal giving the percentage efficiency of the PV panel between 0 (0%
            efficiency), and 1 (100% efficiency).

        """

        # * Determine the electrical efficiency of the panel and return it as a float.


class PVT:
    """
    Represents an entire PV-T collector.

    """

    # Private attributes:
    #
    # .. attribute:: _glass
    #   Represents the upper (glass) layer of the panel.
    #
    # .. attribute:: _pv
    #   Represents the middle (pv) layer of the panel. Can be set to `None` if not
    #   present in the panel.
    #
    # .. attribute:: _collector
    #   Represents the lower (thermal-collector) layer of the panel.
    #
    # .. attribute:: _vertical_tracking
    #   A `bool` giving whether or not the panel tracks verticallly.
    #
    # .. attribute:: _tilt
    #   The angle between the normal to the panel's surface and the horizontal.
    #
    # .. attribute:: _horizontal_tracking
    #   A `bool` giving whether or not the panel tracks horizontally.
    #
    # .. attribute:: _azimuthal_orientation
    #   The angle between the normal to the panel's surface and True North.
    #

    def __init__(
        self,
        tilt: float,
        azimuthal_orientation: float,
        glass_mass: float,
        glass_specific_heat: float,
        collector_mass: float,
        collector_specific_heat: float,
        pv_layer_included: bool = False,
        pv_mass: Optional[float] = None,
        pv_specific_heat: Optional[float] = None,
        horizontal_tracking: bool = False,
        vertical_tracking: bool = False,
    ) -> None:
        """
        Instantiate an instance of the PV-T collector class.

        """

        # * If the PV layer parameters have not been specified, then raise an error.

        if pv_layer_included and (pv_mass is None or pv_specific_heat is None):
            raise MissingParametersError(
                "PVT",
                "PV mass and Specific Heat must be provided if including a PV layer.",
            )

        self._glass = Glass(glass_mass, glass_specific_heat)
        if pv_layer_included:
            self._pv = PV(pv_mass, pv_specific_heat)
        else:
            self._pv = None
        self._collector = Collector(collector_mass, collector_specific_heat)
        self._vertical_tracking = vertical_tracking
        self._tilt = tilt
        self._horizontal_tracking = horizontal_tracking
        self._azimuthal_orientation = azimuthal_orientation

    def _get_solar_angle(
        self, declination: float, azimuthal_angle: float, time_of_day: time.struct_time
    ) -> float:
        """
        Determine the between the panel's normal and the sun.

        :param declination:
            The current angle of declination of the sun in the sky above the horizon.

        :param azimuthal_angle:
            The current azimuthal angle, measured relative to True North, of the sun in
            the sky.

        :param time_of_day:
            Gives the current time of day.

        :return:
            The angle in degrees between the solar irradiance and the normal to the
            panel.

        """

        # * Determine the angle between the sun and panel's normal, both horizontally
        # * and vertically. If tracking is enabled, then this angle should be zero along
        # * each of those axes. If tracking is disabled, then this angle is just the
        # * difference between the panel's orientation and the sun's.

        # * Combine these to generate the angle between the two directions.

    def update(
        self, input_water_temperature: float, weather_conditions: WeatherConditions
    ) -> None:
        """
        Updates the properties of the PV-T collector based on a changed input temp..

        :param input_water_temperature:
            The water temperature going into the PV-T collector.

        :param weather_conditions:
            The weather conditions at the time of day being incremented to.

        """

        # * Compute the angle of solar irradiance wrt the panel

        # * Call the pv panel to update its temperature

        # * Pass this new temperature through to the glass instance to update it

        # * Pass this new temperature through to the collector instance to update it
