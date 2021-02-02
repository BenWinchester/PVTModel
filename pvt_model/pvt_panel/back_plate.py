#!/usr/bin/python3.7
########################################################################################
# pvt_panel/back_plate.py - Represents the back plate of a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The back-plate module for the PV-T model.

This module represents the back plate of a PV-T panel.

"""

from .__utils__ import Layer
from ..__utils__ import (
    BackLayerParameters,
    FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR,
    LayerParameters,
)

__all__ = ("BackPlate",)


class BackPlate(Layer):
    """
    Represents the back-plate layer of the PV-T panel.

    .. attribute:: conductance
        The conducance, measured in Watts per meter squared Kelvin, of the back layer to
        the surroundings.

    """

    def __init__(self, back_params: BackLayerParameters) -> None:
        """
        Instantiate a back layer instance.

        :param back_params:
            The parameters needed to instantiate the back layer of the panel.

        """

        super().__init__(
            LayerParameters(
                back_params.mass,
                back_params.heat_capacity,
                back_params.area,
                back_params.thickness,
                back_params.temperature,
            )
        )

        self.conductivity = back_params.conductivity

    def __repr__(self) -> str:
        """
        Returns a nice representation of the layer.

        :return:
            A `str` giving a nice representation of the layer.

        """
        return "BackPlate(area: {}, conductivity: {}, heat_capacity: {}, mass: {}, ".format(
            self.area,
            self.conductivity,
            self._heat_capacity,  # [J/kg*K]
            self._mass,  # [J/kg*K]
        ) + "temp: {}, thickness: {})".format(
            self.temperature,
            self.thickness,
        )

    @property
    def conductance(self) -> float:
        """
        Returns the conductance of the back plate in Watts per meters squared Kelvin.

        :return:
            The conductance of the layer, measured in Watts per meter squared Kelvin.

        """

        return (
            self.thickness / self.conductivity  # [m] / [W/m*K]
            + 1 / FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR  # [W/m^2*K]^-1
        )

    @property
    def heat_capacity(self) -> float:
        """
        Return the heat capacity of the back plate.

        :return:
            The heat capacity of the back plate layer, measured in Joules per kilogram
            Kelvin.

        """

        return self._heat_capacity

    @property
    def mass(self) -> float:
        """
        Return the mass of the back plate.

        :return:
            The mass of the back plate layer, measured in kilograms.

        """

        return self._mass
