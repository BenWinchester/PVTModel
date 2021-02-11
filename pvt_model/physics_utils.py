#!/usr/bin/python3.7
########################################################################################
# physics_utils.py - Utility module containing Physics equations.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The physics utility module for the PVT model.

"""

from typing import Optional

import numpy

from .constants import (
    STEFAN_BOLTZMAN_CONSTANT,
    THERMAL_CONDUCTIVITY_OF_AIR,
)

from .__utils__ import (
    ProgrammerJudgementFault,
)

__all__ = (
    "conductive_heat_transfer_coefficient_with_gap",
    "convective_heat_transfer_to_fluid",
    "radiative_heat_transfer_coefficient",
    "transmissivity_absorptivity_product",
)


def conductive_heat_transfer_coefficient_with_gap(
    air_gap_thickness: float,
) -> float:
    """
    Computes the conductive heat transfer between the two layers, measured in W/m^2*K.

    The value computed is positive if the heat transfer is from the source to the
    destination, as determined by the arguments, and negative if the flow of heat is
    the reverse of what is implied via the parameters.

    The value for the heat transfer is returned in Watts.

    :param air_gap_thickness:
        The thickness of the air gap between the PV and glass layers.

    :param contact_area:
        The area of contact between the two layers over which conduction can occur,
        measured in meters squared.

    :return:
        The heat transfer coefficient, in Watts per meter squared Kelvin, between the
        two layers.

    """

    return THERMAL_CONDUCTIVITY_OF_AIR / air_gap_thickness  # [W/m*K] / [m]


def convective_heat_transfer_to_fluid(
    contact_area: float,
    convective_heat_transfer_coefficient: float,
    fluid_temperature: float,
    wall_temperature: float,
) -> float:
    """
    Computes the convective heat transfer to a fluid in Watts.

    :param contact_area:
        The surface area that the fluid and solid have in common, i.e., for which they
        are in thermal contact, measured in meters squared.

    :param convective_heat_transfer_coefficient:
        The convective heat transfer coefficient of the fluid, measured in Watts per
        meter squared Kelvin.

    :param fluid_temperature:
        The temperature of the fluid, measured in Kelvin.

    :param wall_temperature:
        The temperature of the walls of the container or pipe surrounding the fluid.

    :return:
        The convective heat transfer to the fluid, measured in Watts. If the value is
        positive, then the heat flow is from the container walls to the fluid. If the
        value returned is negative, then the flow is from the fluid to the container
        walls.

    """

    return (
        convective_heat_transfer_coefficient  # [W/m^2*K]
        * contact_area  # [m^2]
        * (wall_temperature - fluid_temperature)  # [K]
    )


def radiative_heat_transfer_coefficient(
    *,
    destination_emissivity: Optional[float] = None,
    destination_temperature: float,
    radiating_to_sky: Optional[bool] = False,
    source_emissivity: float,
    source_temperature: float,
) -> float:
    """
    Computes the radiative heat transfer coefficient between two layers.

    The value computed should always be positive, and any negative flow is computed when
    the net temperature difference is applied to the value returned by this function.

    @@@ BEN-TO-FIX
    The coefficient is computed by using the difference of two squares. This introduces
    an inaccuracy in the model whereby the coefficient is computed at the previous time
    step dispite depending on the temperatures.

    The value for the heat transfer is returned in Watts per meter squared Kelvin.

    :param destination_emissivity:
        The emissivity of the layer that is receiving the radiation, defined between 0
        and 1. This parameter can be set to `None` in instances where the destination
        has no well-defined emissivity, such as when radiating to the sky.

    :param destination_temperature:
        The temperature of the destination layer/material, measured in Kelvin.

    :param radiating_to_sky:
        Specifies whether the source of the radiation is the sky (True).

    :param source_temperature:
        The temperature of the source layer/material, measured in Kelvin.

    :param source_emissivity:
        The emissivity of the layer that is radiating, defined between 0 and 1.

    :return:
        The heat transfer coefficient, in Watts per meter squared Kelvin, between two
        layers, or from a layer to the sky, that takes place by radiative transfer.

    """

    if radiating_to_sky:
        return (
            STEFAN_BOLTZMAN_CONSTANT  # [W/m^2*K^4]
            * source_emissivity
            * (source_temperature ** 2 + destination_temperature ** 2)  # [K^2]
            * (source_temperature + destination_temperature)  # [K]
        )

    if destination_emissivity is None:
        raise ProgrammerJudgementFault(
            "If radiating to an object that is not the sky, a destination emissivity "
            "must be specified."
        )

    return (
        STEFAN_BOLTZMAN_CONSTANT  # [W/m^2*K^4]
        * (source_temperature ** 2 + destination_temperature ** 2)  # [K^2]
        * (source_temperature + destination_temperature)  # [K]
    ) / ((1 / source_emissivity) + (1 / destination_emissivity) - 1)


def transmissivity_absorptivity_product(
    *,
    diffuse_reflection_coefficient: float,
    glass_transmissivity: float,
    layer_absorptivity: float,
) -> float:
    """
    Computes the transmissivity-absorptivity product for a layer.

    Due to diffuse reflection at the upper (glass) layer of a PVT panel, along with the
    effects of only partial transmission through the layer, the transmissivity-
    absorptivity product for the layer depends on the transmissivity of the layer
    above, as well as the absorptivity of the layer in questiopn, along nwith the
    diffuse reflectivity coefficient.

    :param diffuse_reflection_coefficient:
        The diffuse reflectivity coefficient.

    :param glass_transmissivity:
        The transmissivity of the upper glass layer.

    :param layer_absorptivity:
        The absorptivity of the layer taking in the sunlight.

    :return:
        The transmissivity-absorptivity product for light being absorbed by the layer.

    """

    return (layer_absorptivity * glass_transmissivity) / (
        1 - (1 - layer_absorptivity) * diffuse_reflection_coefficient
    )
