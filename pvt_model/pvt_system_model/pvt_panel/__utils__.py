#!/usr/bin/python3.7
########################################################################################
# pvt_panel/__utils__.py - The utility module for the PVT panel component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The utility module for the PV-T panel component.

This module contains common functionality, strucutres, and types, to be used by the
various modules throughout the PVT panel component.

"""

from ..__utils__ import (
    LayerParameters,
    OpticalLayerParameters,
)

__all__ = (
    "Layer",
    "OpticalLayer",
)


class Layer:
    """
    Represents a layer within the PV-T panel.

    .. attribute:: area
        The area of the layer, measured in meters squared.

    .. attribute:: conductivity
        The conductivity of the layer, measured in Watts per meter Kelvin.

    .. attribute:: density
        The desntiy of the layer, measured in kilograms per meter cubed.

    .. attribute:: heat_capacity
      The heat capacity of the layer, measured in Joules per kilogram Kelvin.

    .. attribute:: mass
      The mass of the layer in kilograms.

    .. attribute:: thickenss
        The thickness (depth) of the layer, measured in meters.

    """

    def __init__(self, layer_params: LayerParameters) -> None:
        """
        Instantiate an instance of the layer class.

        """

        self.area = layer_params.area
        self.conductivity = layer_params.conductivity
        self.density = layer_params.density
        self.heat_capacity = layer_params.heat_capacity  # [J/kg*K]
        self.mass = layer_params.mass  # [kg]
        self.thickness = layer_params.thickness


class OpticalLayer(Layer):
    """
    Represents a layer within the PV-T panel that has optical properties.

    .. attribute:: emissivity
        The emissivity of the layer; a dimensionless number between 0 (nothing is
        emitted by the layer) and 1 (the layer re-emits all incident light).

    """

    # Private Attributes:
    #
    # .. attribute:: _absorptivity
    #   The absorptivity of the layer: a dimensionless number between 0 (nothing is
    #   absorbed by the layer) and 1 (all light is absorbed).
    #

    # .. attribute:: _transmissivity
    #   The transmissivity of the layer: a dimensionless number between 0 (nothing is
    #   transmitted through the layer) and 1 (all light is transmitted).
    #

    def __init__(self, optical_params: OpticalLayerParameters) -> None:
        """
        Instantiate an optical layer within the PV-T panel.

        :param optical_params:
            Contains parameters needed to instantiate the optical layer.

        """

        super().__init__(
            LayerParameters(
                optical_params.area,
                optical_params.conductivity,
                optical_params.density,
                optical_params.heat_capacity,
                optical_params.mass,  # [kg]
                optical_params.thickness,
            )
        )

        self.absorptivity = optical_params.absorptivity
        self.transmissivity = optical_params.transmissivity
        self.emissivity = optical_params.emissivity

    def __repr__(self) -> str:
        """
        Returns a nice representation of the layer.

        :return:
            A `str` giving a nice representation of the layer.

        """

        return (
            "OpticalLayer("
            f"absorptivity: {self.absorptivity}, "
            f"area: {self.area}, "
            f"emissivitiy: {self.emissivity}, "
            f"heat_capacity: {self.heat_capacity}, "
            f"mass: {self.mass}, "
            f"thickness: {self.thickness}, "
            f"transmissivity: {self.transmissivity}"
            ")"
        )
