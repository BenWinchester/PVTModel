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

from .__utils__ import OpticalLayer, OpticalLayerParameters

__all__ = ("Glass",)


class Glass(OpticalLayer):
    """
    Represents the glass (upper) layer of the PV-T panel.

    """

    def __init__(
        self,
        diffuse_reflection_coefficient: float,
        optical_layer_params: OpticalLayerParameters,
    ) -> None:
        """
        Instantiate a glass layer instance.

        :param diffuse_reflection_coefficient:
            The coefficient of diffuse reflectivity of the layer.

        :param optical_layer_params:
            Parameters used to instantiate a generic optical layer.

        """

        self.diffuse_reflection_coefficient = diffuse_reflection_coefficient
        super().__init__(optical_layer_params)
