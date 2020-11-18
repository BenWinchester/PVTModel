#!/usr/bin/python3.7
########################################################################################
# __utils__.py - The utility module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The utility module for the PV-T model.

This module contains common functionality, strucutres, and types, to be used by the
various modules throughout the PVT model.

"""

import os

from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

__all__ = (
    "MissingParametersError",
    "CollectorParameters",
    "LayerParameters",
    "OpticalLayerParameters",
    "PVParameters",
    "WeatherConditions",
    "read_yaml",
    "HEAT_CAPACITY_OF_WATER",
)


#############
# Constants #
#############


# The Stefan-Boltzman constant, given in Watts per meter squared Kelvin to the four.
STEFAN_BOLTZMAN_CONSTANT: float = 5.670374419 * 10 ** (-8)

# The heat capacity of water, measured in Joules per kilogram Kelvin.
HEAT_CAPACITY_OF_WATER: int = 4182

# The free convective heat transfer coefficient of air. This varies, and potentially
# could be extended to the weather module and calculated on the fly depending on various
# environmental conditions etc..
FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR: int = 25

# The thermal conductivity of air is measured in Watts per meter Kelvin.
# ! This is defined at 273 Kelvin.
THERMAL_CONDUCTIVITY_OF_AIR: float = 0.024

# The temperature of absolute zero in Kelvin, used for converting Celcius to Kelvin and
# vice-a-versa.
ZERO_CELCIUS_OFFSET: float = 273.15


##############
# Exceptions #
##############


class MissingParametersError(Exception):
    """
    Raised when not all parameters have been specified that are needed to instantiate.

    """

    def __init__(self, class_name, message) -> None:
        """
        Instantiate a missing parameters error.

        :param class_name:
            The class for which insufficient parameters were specified.

        :param message:
            An appended message to display to the user.

        """

        super().__init__(
            f"Missing parameters when initialising a '{class_name}' class: {message}."
        )


##############################
# Functions and Data Classes #
##############################


@dataclass
class LayerParameters:
    """
    Contains parameters needed to instantiate a layer within the PV-T panel.

    .. attribute:: mass
        The mass of the layer, measured in Kelvin.

    .. attribute:: heat_capacity
        The heat capacity of the layer, measured in Joules per kilogram Kelvin.

    .. attribute:: area
        The area of the layer, measured in meters squared.

    .. attribute:: thickness
        The thickness of the layer, measured in meters.

    .. attribute:: temperature
        The temperature at which to initialise the layer, measured in Kelvin.

    """

    mass: float
    heat_capacity: float
    area: float
    thickness: float
    temperature: Optional[float]


@dataclass
class CollectorParameters(LayerParameters):
    """
    Contains parameters needed to instantiate a collector layer within the PV-T panel.

    .. attribute:: output_water_temperature
        The temperature, in Kelvin, of water outputted by the layer.

    .. attribute:: mass_flow_rate
        The mass flow rate of heat-transfer fluid through the collector.

    """

    output_water_temperature: float
    mass_flow_rate: float


@dataclass
class BackLayerParameters(LayerParameters):
    """
    Contains parameters needed to instantiate the back layer of the PV-T panel.

    .. attribute:: conductance
        The conductance of layer (to the environment/its surroundings), measured in
        Watts per meter squared Kelvin.

    """

    conductivity: float


@dataclass
class OpticalLayerParameters(LayerParameters):
    """
    Contains parameters needed to instantiate a layer with optical properties.

    .. attribute:: transmissivity
        The transmissivity of the layer: a dimensionless number between 0 (nothing is
        transmitted through the layer) and 1 (all light is transmitted).

    .. attribute:: absorptivity
        The absorptivity of the layer: a dimensionless number between 0 (nothing is
        absorbed by the layer) and 1 (all light is absorbed).

    .. attribute:: emissivity
        The emissivity of the layer; a dimensionless number between 0 (nothing is
        emitted by the layer) and 1 (the layer re-emits all incident light).

    """

    transmissivity: float
    absorptivity: float
    emissivity: float


@dataclass
class PVParameters(OpticalLayerParameters):
    """
    Contains parameters needed to instantiate a PV layer within the PV-T panel.

    .. attribute:: reference_efficiency
        The efficiency of the PV layer at the reference temperature. Thie value varies
        between 1 (corresponding to 100% efficiency), and 0 (corresponding to 0%
        efficiency)

    .. attribute:: reference_temperature
        The referencee temperature, in Kelvin, at which the reference efficiency is
        defined.

    .. attribute:: thermal_coefficient
        The thermal coefficient for the efficiency of the panel.

    """

    reference_efficiency: float
    reference_temperature: float
    thermal_coefficient: float


@dataclass
class WeatherConditions:
    """
    Contains information about the various weather conditions at any given time.

    .. attribute:: irradiance
        The solar irradiance in Watts per meter squared.

    .. attribute:: declination
        The angle of declination of the sun above the horizon

    .. attribute:: azimuthal_angle
        The azimuthal angle of the sun, defined clockwise from True North.

    .. attribute:: wind_speed
        The wind speed in meters per second.

    .. attribute:: ambient_temperature
        The ambient temperature in

    """

    irradiance: float
    declination: float
    azimuthal_angle: float
    wind_speed: float
    ambient_temperature: float

    @property
    def sky_temperature(self) -> float:
        """
        Determines the radiative temperature of the sky.

        The "sky," as a black body, has a radiative temperature different to that of the
        surrounding air, or the ambient temperature. This function converts between them
        and outputs the sky's radiative temperature.

        :return:
            The radiative temperature of the "sky" in Kelvin.

        """

        return 0.0552 * (self.ambient_temperature ** 1.5)


def read_yaml(yaml_file_path: str) -> Dict[Any, Any]:
    """
    Read in some yaml data and return it.

    :param yaml_file_path:
        The path to the yaml data to read in.

    :return:
        A `dict` containing the data read in from the yaml file.

    """

    # Open the yaml data and read it.
    if not os.path.isfile(yaml_file_path):
        raise yaml_file_path("")
    with open(yaml_file_path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.parser.ParserError as e:
            # * Do some logging
            raise

    return data
