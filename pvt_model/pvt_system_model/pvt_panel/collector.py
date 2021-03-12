#!/usr/bin/python3.7
########################################################################################
# pvt_panel/collector.py - Represents a collector within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The collector module for the PV-T model.

This module represents a thermal collector within a PV-T panel.

"""

import logging
import math

from ..__utils__ import (
    CollectorParameters,
    PVT_SYSTEM_MODEL_LOGGER_NAME,
    OpticalLayerParameters,
)
from ..constants import THERMAL_CONDUCTIVITY_OF_WATER
from .__utils__ import (
    OpticalLayer,
)

__all__ = ("Collector",)

# Get the logger for the run.
logger = logging.getLogger(PVT_SYSTEM_MODEL_LOGGER_NAME)


class Collector(OpticalLayer):
    """
    Represents the thermal collector (lower) layer of the PV-T panel.

    .. attribute:: htf_heat_capacity
        The heat capacity of the heat-transfer fluid passing through the collector,
        measured in Joules per kilogram Kelvin.

    .. attribute:: mass_flow_rate
        The mass flow rate of heat-transfer fluid through the collector, measured in
        kilograms per second.

    .. attribute:: output_water_temperature
        The temperature of the water outputted by the layer, measured in Kelvin.

    .. attribute:: pump_power
        The power consumed by the water pump, measured in Watts.

    """

    # Pirvate Attributes:
    #
    # .. attribute:: _mass_flow_rate
    #   The mass flow rate of heat-trasnfer fluid through the collector, measured in
    #   Litres per hour.
    #

    def __init__(self, collector_params: CollectorParameters) -> None:
        """
        Instantiate a collector layer.

        :param collector_params:
            The parameters needed to instantiate the collector.

        """

        super().__init__(
            OpticalLayerParameters(
                collector_params.conductivity,
                collector_params.density,
                collector_params.heat_capacity,
                collector_params.thickness,
                collector_params.transmissivity,
                collector_params.absorptivity,
                collector_params.emissivity,
            )
        )

        self.htf_heat_capacity = collector_params.htf_heat_capacity
        self.inner_pipe_diameter = collector_params.inner_pipe_diameter
        self.length = collector_params.length
        self._mass_flow_rate = collector_params.mass_flow_rate
        self.number_of_pipes = collector_params.number_of_pipes
        self.outer_pipe_diameter = collector_params.outer_pipe_diameter

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
    def convective_heat_transfer_coefficient_of_water(self) -> float:
        """
        Returns the convective heat transfer coefficient of water, measured in W/m^2*K.

        :return:
            The convective heat transfer coefficient of water, calculated from the
            Nusselt number for the flow, the conductivity of water, and the pipe
            diameter.

        """

        h_f: float = 4.36 * THERMAL_CONDUCTIVITY_OF_WATER / self.inner_pipe_diameter

        return h_f

        # return 1000 * h_f

    @property
    def htf_surface_area(self) -> float:
        """
        Returns the contact area between the HTF and the collector, measured in m^2.

        :return:
            The contact surface area, between the collector (i.e., the pipes) and the
            HTF passing through the pipes.
            A single pass is assumed, with multiple pipes increasing the area, rather
            than the length, of the collector.

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
        Returns the volume of HTF that can be held within the collector, measured in m^3

        :return:
            The volume of the HTF within the collector, measured in meters cubed.

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
