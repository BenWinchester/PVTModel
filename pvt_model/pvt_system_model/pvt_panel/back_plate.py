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
    LayerParameters,
)
from ..constants import FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR

__all__ = ("BackPlate",)


class BackPlate(Layer):
    """
    Represents the back-plate layer of the PV-T panel.

    .. attribute:: conductance
        The conducance, measured in Watts per meter squared Kelvin, of the back layer to
        the surroundings.

    """

    @property
    def conductance(self) -> float:
        """
        Returns the conductance of the back plate in Watts per meters squared Kelvin.

        :return:
            The conductance of the layer, measured in Watts per meter squared Kelvin.

        """

        return (
            self.thickness / self.conductivity  # [m] / [W/m*K]
            + 1 / FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR  # [m^2*K/W]
        ) ** (
            -1
        )  # [W/m^2*K]
