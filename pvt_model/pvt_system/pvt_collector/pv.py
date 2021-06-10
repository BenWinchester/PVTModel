#!/usr/bin/python3.7
########################################################################################
# pvt_collector/pv.py - Represents a absorber within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The PV module for the PV-T model.

This module represents a PV layer within a PV-T panel.

"""

from ..__utils__ import (
    OpticalLayerParameters,
    PVParameters,
)
from .__utils__ import (
    OpticalLayer,
)

__all__ = ("PV",)


class PV(OpticalLayer):
    """
    Represents the photovoltaic (middle) layer of the PV-T panel.

    """

    # Private attributes:
    #
    # .. attribute:: reference_efficiency
    #   The efficiency of the PV layer at the reference temperature. Thie value varies
    #   between 1 (corresponding to 100% efficiency), and 0 (corresponding to 0%
    #   efficiency)
    #
    # .. attribute:: reference_temperature
    #   The referencee temperature, in Kelvin, at which the reference efficiency is
    #   defined.
    #
    # .. attribute:: thermal_coefficient
    #   The thermal coefficient for the efficiency of the panel.
    #

    def __init__(self, pv_params: PVParameters) -> None:
        """
        Instantiate a PV layer.

        :param pv_params:
            Parameters needed to instantiate the PV layer.

        """

        super().__init__(
            OpticalLayerParameters(
                pv_params.conductivity,
                pv_params.density,
                pv_params.heat_capacity,
                pv_params.thickness,
                pv_params.transmissivity,
                pv_params.absorptivity,
                pv_params.emissivity,
            )
        )

        self.reference_efficiency = pv_params.reference_efficiency
        self.reference_temperature = pv_params.reference_temperature
        self.thermal_coefficient = pv_params.thermal_coefficient

    def __repr__(self) -> str:
        """
        Returns a nice representation of the layer.

        :return:
            A `str` giving a nice representation of the layer.

        """

        return (
            "PV("
            f"absorptivity: {self.absorptivity}, "
            f"heat_capacity: {self.heat_capacity}J/kg*K, "
            f"transmissivity: {self.transmissivity}, "
            f"emissivity: {self.emissivity}, "
            f"reference_efficiency: {self.reference_efficiency}, "
            f"reference_temperature: {self.reference_temperature}K, "
            f"reflectivity: {self.reflectivity:.3f}, "
            f"thermal_coefficient: {self.thermal_coefficient}K^(-1), "
            f"thickness: {self.thickness}m"
            ")"
        )

    def electrical_efficiency(self, pv_temperature: float) -> float:
        """
        Returns the electrical efficiency of the PV panel based on its temperature.

        :param pv_temperature:
            The temperature of the PV layer, measured in Kelvin.

        :return:
            A decimal giving the percentage efficiency of the PV panel between 0 (0%
            efficiency), and 1 (100% efficiency).

        """

        return self.reference_efficiency * (  # [unitless]
            1
            - self.thermal_coefficient  # [1/K]
            * (pv_temperature - self.reference_temperature)  # [K]
        )
