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

from typing import List, Optional, Tuple, Union

import math

from numpy import ndarray, pi

from .constants import (
    ACCELERATION_DUE_TO_GRAVITY,
    STEFAN_BOLTZMAN_CONSTANT,
)
from .pvt_panel import pvt
from .pvt_panel.segment import Segment

from .__utils__ import ProgrammerJudgementFault, WeatherConditions

__all__ = (
    "convective_heat_transfer_to_fluid",
    "free_heat_transfer_coefficient_of_air",
    "grashof_number",
    "prandtl_number",
    "radiative_heat_transfer_coefficient",
    "rayleigh_number",
    "reduced_temperature",
    "upward_loss_terms",
)


def _top_heat_transfer_coefficient(
    pvt_panel: pvt.PVT,
    segment_top_temperature: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the heat-transfer coeffient between the top of the panel and the air.

    NOTE: This includes both conductive (free) and convective (forced) heat transfers.

    :param pvt_panel:
        The pvt panel being modelled.

    :param segment_top_temperature:
        The temperature, measured in Kelvin, of the top-layer of the segment.

    :param weather_conditions:
        The weather conditions at the time step being modelled.

    :return:
        The heat transfer coefficient between the panel and the air, measured in Watts
        per meter squared Kelvin.

    """

    heat_transfer_coefficient: float = (
        weather_conditions.wind_heat_transfer_coefficient ** 3
        + free_heat_transfer_coefficient_of_air(
            pvt_panel, segment_top_temperature, weather_conditions
        )
        ** 3
    ) ** (1 / 3)

    return heat_transfer_coefficient


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


def free_heat_transfer_coefficient_of_air(
    pvt_panel: pvt.PVT,
    segment_top_temperature: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the free (conductive/convective) heat-transfer coefficient of air.

    :param pvt_panel:
        The pvt panel being modelled.

    :param segment_top_temperature:
        The temperature, measured in Kelvin, of the top-layer of the segment.

    :param weather_conditions:
        The weather conditions at the time step being modelled.

    :return:
        The free heat transfer coefficient between the panel and the air, measured in
        Watts per meter squared Kelvin.

    """

    length_scale: float = max(pvt_panel.length, pvt_panel.width)

    current_rayleigh_number = rayleigh_number(
        length_scale,
        segment_top_temperature,
        pvt_panel.tilt_in_radians,
        weather_conditions,
    )

    if current_rayleigh_number >= 10 ** 9:
        return (weather_conditions.thermal_conductivity_of_air / length_scale) * (
            0.68
            + (
                (0.67 * current_rayleigh_number ** 0.25)
                / (
                    (1 + (0.492 / prandtl_number(weather_conditions)) ** (9 / 16))
                    ** (4 / 9)
                )
            )
        )

    return (weather_conditions.thermal_conductivity_of_air / length_scale) * (
        nusselt_number(
            length_scale,
            segment_top_temperature,
            pvt_panel.tilt_in_radians,
            weather_conditions,
        )
        ** (1 / 4)
    )


def grashof_number(
    length_scale: float,
    surface_temperature: float,
    tilt_in_radians: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the dimensionless Grashof number.

    :param length_scale:
        The length scale over which the Grashof number should be computed.

    :param surface_temperature:
        The temperature of the bluff surface for which the Grashof number should be
        computed.

    :param tilt_in_radians:
        The tilt, in radians, between the panel (surface) and the horizontal.

    :param weather_conditions:
        The weather conditions at the time step where the Grashof number should be
        computed.

    :return:
        The dimensionless Grashof number.

    """

    return (
        weather_conditions.thermal_expansivity_of_air  # [1/K]
        * ACCELERATION_DUE_TO_GRAVITY  # [m/s^2]
        * math.cos((pi / 2) - tilt_in_radians)
        * weather_conditions.density_of_air ** 2  # [kg/m^3]
        * length_scale ** 3  # [m^3]
        * abs(surface_temperature - weather_conditions.ambient_temperature)  # [K]
    ) / (
        weather_conditions.dynamic_viscosity_of_air ** 2
    )  # [kg/m*s]


def nusselt_number(
    length_scale: float,
    surface_temperature: float,
    tilt_in_radians: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the dimensionless Nusselt number.

    :param length_scale:
        The length scale over which the Grashof number should be computed.

    :param surface_temperature:
        The temperature of the bluff surface for which the Grashof number should be
        computed.

    :param tilt_in_radians:
        The tilt, in radians, between the panel (surface) and the horizontal.

    :param weather_conditions:
        The weather conditions at the time step where the Grashof number should be
        computed.

    :return:
        The dimensionless Nusselt number.

    """

    return 0.825 + (
        0.387
        * rayleigh_number(
            length_scale, surface_temperature, tilt_in_radians, weather_conditions
        )
        ** (1 / 6)
    ) / ((1 + (0.492 / prandtl_number(weather_conditions) ** (9 / 16))) ** (8 / 27))


def prandtl_number(weather_conditions: WeatherConditions) -> float:
    """
    Computes the dimensionless Prandtl number.

    :param weather_conditions:
        The weather conditions at the time step where the Prandtl number should be
        computed.

    :return:
        The dimensionless Prandtl number.

    """

    return (
        weather_conditions.heat_capacity_of_air
        * weather_conditions.dynamic_viscosity_of_air
        / weather_conditions.thermal_conductivity_of_air
    )


def radiative_heat_transfer_coefficient(
    *,
    destination_temperature: float,
    source_emissivity: float,
    source_temperature: float,
    destination_emissivity: Optional[float] = None,
    radiating_to_sky: Optional[bool] = False,
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

    :param destination_temperature:
        The temperature of the destination layer/material, measured in Kelvin.

    :param source_temperature:
        The temperature of the source layer/material, measured in Kelvin.

    :param source_emissivity:
        The emissivity of the layer that is radiating, defined between 0 and 1.

    :param destination_emissivity:
        The emissivity of the layer that is receiving the radiation, defined between 0
        and 1. This parameter can be set to `None` in instances where the destination
        has no well-defined emissivity, such as when radiating to the sky.

    :param radiating_to_sky:
        Specifies whether the source of the radiation is the sky (True).

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


def rayleigh_number(
    length_scale: float,
    surface_temperature: float,
    tilt_in_radians: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the non-dimensional Rayleigh number.

    :param length_scale:
        The length scale over which the Grashof number should be computed.

    :param surface_temperature:
        The temperature of the bluff surface for which the Grashof number should be
        computed.

    :param tilt_in_radians:
        The tilt, in radians, between the panel (surface) and the horizontal.

    :param weather_conditions:
        The weather conditions at the time step where the Grashof number should be
        computed.

    :return:
        The dimensionless Grashof number.

    """

    return grashof_number(
        length_scale, surface_temperature, tilt_in_radians, weather_conditions
    ) * prandtl_number(weather_conditions)


def reduced_temperature(
    ambient_temperature: float, average_temperature: float, solar_irradiance: float
) -> float:
    """
    Computes the reduced temperature of the collector.

    NOTE: The ambient temperature and average temperature need to be measured in the
    same units, whether it's Kelvin or Celcius, but it does not matter which of these
    two is used.

    :param ambient_temperature:
        The ambient temperature surrounding the collector.

    :param average_temperature:
        The average temperature of the collector.

    :param solar_irradiance:
        The solar irradiance, measured in Watts per meter squared.

    :return:
        The reduced temperature of the collector in Kelvin meter squared per Watt.

    """

    return (average_temperature - ambient_temperature) / solar_irradiance


def upward_loss_terms(
    best_guess_temperature_vector: Union[List[float], ndarray],
    pvt_panel: pvt.PVT,
    segment: Segment,
    source_index: int,
    weather_conditions: WeatherConditions,
) -> Tuple[float, float]:
    """
    Computes the upward convective/conductive and radiative loss terms.

    :param best_guess_temperature_vector:
        The best guess at the temperature vector at the current time step.

    :param pvt_panel:
        The pvt panel being modelled.

    :param segment:
        The segment currently being considered.

    :param source_index:
        The index of the source segment.

    :param weather_conditions:
        The weather conditions at the time step being modelled.

    :return:
        A `tuple` containing:
        - the convective/conductive heat transfer upward from the layer,
        - the radiative heat transfer upward from the layer.

    """

    upward_conduction = (
        segment.width  # [m]
        * segment.length  # [m]
        * _top_heat_transfer_coefficient(
            pvt_panel, best_guess_temperature_vector[source_index], weather_conditions
        )  # [W/m^2*K]
    )

    upward_radiation = (
        segment.width  # [m]
        * segment.length  # [m]
        * radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=best_guess_temperature_vector[source_index],
        )  # [W/m^2*K]
    )

    return upward_conduction, upward_radiation
