#!/usr/bin/python3.7
########################################################################################
# pvt_panel/pv.py - Represents a collector within a PVT panel.
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

    def __init__(self, pv_params: PVParameters) -> None:
        """
        Instantiate a PV layer.

        :param pv_params:
            Parameters needed to instantiate the PV layer.

        """

        super().__init__(
            OpticalLayerParameters(
                pv_params.mass,
                pv_params.heat_capacity,
                pv_params.area,
                pv_params.thickness,
                pv_params.transmissivity,
                pv_params.absorptivity,
                pv_params.emissivity,
            )
        )

        self._reference_efficiency = pv_params.reference_efficiency
        self._reference_temperature = pv_params.reference_temperature
        self._thermal_coefficient = pv_params.thermal_coefficient

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
            f"_reference_efficiency: {self._reference_efficiency}, "
            f"_reference_temperature: {self._reference_temperature}K, "
            f"_thermal_coefficient: {self._thermal_coefficient}K^(-1), "
            f"_transmissicity: {self.transmissivity}, "
            f"area: {self.area}m^2, "
            f"emissivity: {self.emissivity}, "
            f"mass: {self.mass}kg, "
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

        return self._reference_efficiency * (  # [unitless]
            1
            - self._thermal_coefficient  # [1/K]
            * (pv_temperature - self._reference_temperature)  # [K]
        )
