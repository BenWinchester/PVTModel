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
    THERMAL_CONDUCTIVITY_OF_WATER,
    ZERO_CELCIUS_OFFSET,
)
from .pvt_collector import pvt
from .pvt_collector.element import Element

from .__utils__ import ProgrammerJudgementFault, WeatherConditions

__all__ = (
    "convective_heat_transfer_coefficient_of_water",
    "density_of_water",
    "dynamic_viscosity_of_water",
    "free_heat_transfer_coefficient_of_air",
    "grashof_number",
    "prandtl_number",
    "radiative_heat_transfer_coefficient",
    "rayleigh_number",
    "reduced_temperature",
    "reynolds_number",
    "upward_loss_terms",
)


def _top_heat_transfer_coefficient(
    pvt_collector: pvt.PVT,
    element_top_temperature: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the heat-transfer coeffient between the top of the panel and the air.

    NOTE: This includes both conductive (free) and convective (forced) heat transfers.

    :param pvt_collector:
        The pvt panel being modelled.

    :param element_top_temperature:
        The temperature, measured in Kelvin, of the top-layer of the element.

    :param weather_conditions:
        The weather conditions at the time step being modelled.

    :return:
        The heat transfer coefficient between the panel and the air, measured in Watts
        per meter squared Kelvin.

    """

    heat_transfer_coefficient: float = (
        weather_conditions.wind_heat_transfer_coefficient ** 3
        + free_heat_transfer_coefficient_of_air(
            pvt_collector, element_top_temperature, weather_conditions
        )
        ** 3
    ) ** (1 / 3)

    return heat_transfer_coefficient


def convective_heat_transfer_coefficient_of_water(
    fluid_temperature: float,
    pvt_collector: pvt.PVT,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the convective heat transfer to a fluid in Watts.

    :param fluid_temperature:
        The temperature of the fluid element, measured in Kelvin.

    :param pvt_collector:
        The PVT panel being modelled.

    :param weather_conditions:
        The weather conditions at the current time step.

    :return:
        The convective heat transfer to the fluid, measured in Watts. If the value is
        positive, then the heat flow is from the container walls to the fluid. If the
        value returned is negative, then the flow is from the fluid to the container
        walls.

    """

    # Convert the mass-flow rate into a flow speed.
    flow_speed: float = pvt_collector.absorber.mass_flow_rate / (
        density_of_water(fluid_temperature)
        * pi
        * (pvt_collector.absorber.inner_pipe_diameter / 2) ** 2
    )

    # Compute the current Reynolds number.
    current_reynolds_number = reynolds_number(
        density_of_water(fluid_temperature),
        dynamic_viscosity_of_water(fluid_temperature),
        flow_speed,
        pvt_collector.absorber.inner_pipe_diameter,
    )

    if flow_speed == 0:
        return (
            2
            * THERMAL_CONDUCTIVITY_OF_WATER
            / pvt_collector.absorber.inner_pipe_diameter
        )

    if current_reynolds_number < 2300:
        return (
            4.36
            * THERMAL_CONDUCTIVITY_OF_WATER
            / pvt_collector.absorber.inner_pipe_diameter
        )

    return (
        0.23
        * (current_reynolds_number ** 0.8)
        * (prandtl_number(weather_conditions) ** 0.4)
        * THERMAL_CONDUCTIVITY_OF_WATER
        / pvt_collector.absorber.inner_pipe_diameter
    )


def density_of_water(fluid_temperature: float) -> float:
    """
    The density of water varies as a function of temperature.

    The formula for the density is obtained from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4909168/

    :param fluid_temperature:
        The temperature of the fluid, measured in Kelvin.

    :return:
        The density of water, measured in kilograms per meter cubed.

    """

    return (
        999.85308
        + 6.32693 * (10 ** (-2)) * (fluid_temperature - ZERO_CELCIUS_OFFSET)
        - 8.523892 * (10 ** (-3)) * (fluid_temperature - ZERO_CELCIUS_OFFSET) ** 2
        + 6.943249 * (10 ** (-5)) * (fluid_temperature - ZERO_CELCIUS_OFFSET) ** 3
        - 3.82126 * (10 ** (-7)) * (fluid_temperature - ZERO_CELCIUS_OFFSET) ** 4
    )


def dynamic_viscosity_of_water(fluid_temperature: float) -> float:
    """
    The dynamic viscosity of water varies as a function of temperature.

    The formula comes from the Vogel-Fulcher-Tammann equation via Wiki:
    https://en.wikipedia.org/wiki/Viscosity#Water

    :param fluid_temperature:
        The temperature of the fluid being modelled, measured in Kelvin.

    :return:
        The dynamic viscosity of water, measured in kilograms per meter second.

    """

    return 0.00002939 * math.exp(507.88 / (fluid_temperature - 149.3))


def free_heat_transfer_coefficient_of_air(
    pvt_collector: pvt.PVT,
    element_top_temperature: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the free (conductive/convective) heat-transfer coefficient of air.

    :param pvt_collector:
        The pvt panel being modelled.

    :param element_top_temperature:
        The temperature, measured in Kelvin, of the top-layer of the element.

    :param weather_conditions:
        The weather conditions at the time step being modelled.

    :return:
        The free heat transfer coefficient between the panel and the air, measured in
        Watts per meter squared Kelvin.

    """

    length_scale: float = max(pvt_collector.length, pvt_collector.width)

    current_rayleigh_number = rayleigh_number(
        length_scale,
        element_top_temperature,
        pvt_collector.tilt_in_radians,
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
            element_top_temperature,
            pvt_collector.tilt_in_radians,
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

    # Some cases need to be dealt with where the emissivity of one or more layers is
    # zero. In these cases, the emissivity of the non-zero layer, if there is one, can
    # be used.
    if destination_emissivity == 0 and source_emissivity == 0:
        return 0
    if destination_emissivity == 0:
        return source_emissivity
    if source_emissivity == 0:
        return destination_emissivity

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


def reynolds_number(
    density: float, dynamic_viscosity: float, flow_speed: float, length_scale: float
) -> float:
    """
    Computes the Reynolds number of the flow.

    :param density:
        The density of the fluid, measured in kilograms per meter cubed.

    :param dynamic_viscosity:
        The dynamic viscosity, measured in kilograms per meter second.

    :param flow_speed:
        The speed of the flow, measured in meters per second.

    :param length_scale:
        A characteristic length scale over which Physics in the fluid is occurring.

    :return:
        The dimensionless Reynolds number.

    """

    return (
        density  # [kg/m^3]
        * flow_speed  # [m/s]
        * length_scale  # [m]
        / dynamic_viscosity  # [kg/m*s]
    )


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
    pvt_collector: pvt.PVT,
    element: Element,
    source_emissivity: float,
    source_index: int,
    weather_conditions: WeatherConditions,
) -> Tuple[float, float]:
    """
    Computes the upward convective/conductive and radiative loss terms.

    :param best_guess_temperature_vector:
        The best guess at the temperature vector at the current time step.

    :param pvt_collector:
        The pvt panel being modelled.

    :param element:
        The element currently being considered.

    :param source_emissivity:
        The emissivity of the upper layer of the element currently being considered.

    :param source_index:
        The index of the source element.

    :param weather_conditions:
        The weather conditions at the time step being modelled.

    :return:
        A `tuple` containing:
        - the convective/conductive heat transfer upward from the layer,
        - the radiative heat transfer upward from the layer.

    """

    upward_conduction = (
        element.width  # [m]
        * element.length  # [m]
        * _top_heat_transfer_coefficient(
            pvt_collector,
            best_guess_temperature_vector[source_index],
            weather_conditions,
        )  # [W/m^2*K]
    )

    upward_radiation = (
        element.width  # [m]
        * element.length  # [m]
        * radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            source_emissivity=source_emissivity,
            source_temperature=best_guess_temperature_vector[source_index],
        )  # [W/m^2*K]
    )

    return upward_conduction, upward_radiation
