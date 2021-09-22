#!/usr/bin/python3.7
########################################################################################
# pvt_collector/absorber.py - Represents a absorber within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The absorber module for the PV-T model.

This module represents a thermal absorber within a PV-T panel.

"""

import logging
import math

from ..__utils__ import (
    CollectorParameters,
    PVT_SYSTEM_MODEL_LOGGER_NAME,
    OpticalLayerParameters,
)
from .__utils__ import (
    OpticalLayer,
)

__all__ = ("Collector",)

# Get the logger for the run.
logger = logging.getLogger(PVT_SYSTEM_MODEL_LOGGER_NAME)


class Collector(OpticalLayer):
    """
    Represents the thermal absorber (lower) layer of the PV-T panel.

    .. attribute:: htf_heat_capacity
        The heat capacity of the heat-transfer fluid passing through the absorber,
        measured in Joules per kilogram Kelvin.

    .. attribute:: mass_flow_rate
        The mass flow rate of heat-transfer fluid through the absorber, measured in
        kilograms per second.

    .. attribute:: output_water_temperature
        The temperature of the water outputted by the layer, measured in Kelvin.

    .. attribute:: pump_power
        The power consumed by the water pump, measured in Watts.

    """

    # Pirvate Attributes:
    #
    # .. attribute:: _mass_flow_rate
    #   The mass flow rate of heat-trasnfer fluid through the absorber, measured in
    #   Litres per hour.
    #

    def __init__(self, absorber_params: CollectorParameters) -> None:
        """
        Instantiate a absorber layer.

        :param absorber_params:
            The parameters needed to instantiate the absorber.

        """

        super().__init__(
            OpticalLayerParameters(
                absorber_params.conductivity,
                absorber_params.density,
                absorber_params.heat_capacity,
                absorber_params.thickness,
                absorber_params.transmissivity,
                absorber_params.absorptivity,
                absorber_params.emissivity,
            )
        )

        self.htf_heat_capacity = absorber_params.htf_heat_capacity
        self.inner_pipe_diameter = absorber_params.inner_pipe_diameter
        self.length = absorber_params.length
        self._mass_flow_rate = absorber_params.mass_flow_rate
        self.number_of_pipes = absorber_params.number_of_pipes
        self.outer_pipe_diameter = absorber_params.outer_pipe_diameter
        self.pipe_density = absorber_params.pipe_density

    def __repr__(self) -> str:
        """
        Returns a nice representation of the layer.

        :return:
            A `str` giving a nice representation of the layer.

        """

        return (
            "Collector("
            f"absorptivity: {self.absorptivity}, "
            f"conductivity: {self.conductivity}W/m^2*K, "
            f"desntiy: {self.density}kg/m^3, "
            f"emissivity: {self.emissivity}, "
            f"heat_capacity: {self.heat_capacity}J/kg*K, "
            f"htf_heat_capacity: {self.htf_heat_capacity}J/kg*K, "
            f"inner_pipe_diameter: {self.inner_pipe_diameter}m, "
            f"length: {self.length}m, "
            f"mass_flow_rate: {self.mass_flow_rate}kg/s, "
            f"outer_pipe_diameter: {self.outer_pipe_diameter}m, "
            f"thickness: {self.thickness}m, "
            f"transmissivity: {self.transmissivity}"
            ")"
        )

    @property
    def htf_surface_area(self) -> float:
        """
        Returns the contact area between the HTF and the absorber, measured in m^2.

        :return:
            The contact surface area, between the absorber (i.e., the pipes) and the
            HTF passing through the pipes.
            A single pass is assumed, with multiple pipes increasing the area, rather
            than the length, of the absorber.

        """

        return (
            self.number_of_pipes  # [pipes]
            * math.pi
            * self.inner_pipe_diameter  # [m]
            * self.length  # [m]
        )

    @property
    def htf_volume(self) -> float:
        """
        Returns the volume of HTF that can be held within the absorber, measured in m^3

        :return:
            The volume of the HTF within the absorber, measured in meters cubed.

        """

        return (
            self.number_of_pipes  # [pipes]
            * math.pi
            * (self.inner_pipe_diameter / 2) ** 2  # [m^2]
            * self.length
        )

    @property
    def mass_flow_rate(self) -> float:
        """
        Return the mass-flow rate in kilograms per second.

        :return:
            d/dt(M) in kg/s

        """

        return self._mass_flow_rate / (3600)  # [kg/s]
