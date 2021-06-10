#!/usr/bin/python3.7
########################################################################################
# pvt_collector/__utils__.py - The utility module for the PVT panel component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The utility module for the PV-T panel component.

This module contains common functionality, strucutres, and types, to be used by the
various modules throughout the PVT panel component.

"""

from dataclasses import dataclass
from math import sqrt

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

    .. attribute:: conductivity
        The conductivity of the layer, measured in Watts per meter Kelvin.

    .. attribute:: density
        The desntiy of the layer, measured in kilograms per meter cubed.

    .. attribute:: heat_capacity
      The heat capacity of the layer, measured in Joules per kilogram Kelvin.

    .. attribute:: thickenss
        The thickness (depth) of the layer, measured in meters.

    """

    def __init__(self, layer_params: LayerParameters) -> None:
        """
        Instantiate an instance of the layer class.

        """

        self.conductivity = layer_params.conductivity
        self.density = layer_params.density
        self.heat_capacity = layer_params.heat_capacity  # [J/kg*K]
        self.thickness = layer_params.thickness


@dataclass
class MicroLayer:
    """
    Represents a layer within the panel which is small enough to ignore dynamic terms.

    Such layers, such as the adhesive between two layers, are too small to be treated in
    the same way as actual layers as their dynamic (heat capacity) terms are too small
    to warrant consideration.

    .. attribute:: conductivity
        The conductivity of the layer, measured in Watts per meter Kelvin.

    .. attribute:: thickenss
        The thickness (depth) of the layer, measured in meters.

    """

    conductivity: float
    thickness: float


class OpticalLayer(Layer):
    """
    Represents a layer within the PV-T panel that has optical properties.

    .. attribute:: absorptivity
        The absorptivity of the layer: a dimensionless number between 0 (nothing is
        absorbed by the layer) and 1 (all light is absorbed).

    .. attribute:: emissivity
        The emissivity of the layer; a dimensionless number between 0 (nothing is
        emitted by the layer) and 1 (the layer re-emits all incident light).

    .. attribute:: transmissivity
        The transmissivity of the layer: a dimensionless number between 0 (nothing is
        transmitted through the layer) and 1 (all light is transmitted).

    """

    def __init__(self, optical_params: OpticalLayerParameters) -> None:
        """
        Instantiate an optical layer within the PV-T panel.

        :param optical_params:
            Contains parameters needed to instantiate the optical layer.

        """

        super().__init__(
            LayerParameters(
                optical_params.conductivity,
                optical_params.density,
                optical_params.heat_capacity,
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
            f"emissivitiy: {self.emissivity}, "
            f"heat_capacity: {self.heat_capacity}, "
            f"reflectivity: {self.reflectivity:.3f}, "
            f"thickness: {self.thickness}, "
            f"transmissivity: {self.transmissivity}"
            ")"
        )

    @property
    def reflectance(self) -> float:
        """
        Returns the reflectance of the layer.

        :return:
            The reflectance of the layer.

        """

        return self.reflectivity ** 2

    @property
    def reflectivity(self) -> float:
        """
        Returns the reflectivity of the layer.

        :return:
            The reflectivity of the layer, based on its other optical properties.

        """

        return sqrt(1 - self.absorptivity ** 2 - self.transmissivity ** 2)

    @property
    def transmittance(self) -> float:
        """
        Returns the transmittance of the layer.

        :return:
            The transmittance of the layer.

        """

        return self.transmissivity ** 2
