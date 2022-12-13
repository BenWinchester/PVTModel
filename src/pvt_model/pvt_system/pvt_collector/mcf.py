#!/usr/bin/python3.7
########################################################################################
# pvt_collector/mcf.py - Represents a absorber within a PVT panel.
#
# Author: Ben Winchester, Maria Rita Golia
# Copyright: Ben Winchester, 2022
########################################################################################

"""
The MCF module for the PV-T model.

This module represents a MCF layer within a PV-T panel.

NOTE: This needs to contain all the properties about the MCF.

"""

from ..__utils__ import (
    OpticalLayerParameters,
    PVParameters,
)
from .__utils__ import (
    OpticalLayer,
)

__all__ = ("PV",)


class MCF(OpticalLayer):
    """
    Represents the photovoltaic (middle) layer of the PV-T panel.

    """

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
