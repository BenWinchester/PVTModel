#!/usr/bin/python3.7
########################################################################################
# pvt_panel/panel_utils.py - A Physics utility module for the PVT panel component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Physics utility module for the PV-T panel component.

This module contains formulae for calculating Physical values in relation to the PVT
panel.

"""

from ..constants import DENSITY_OF_WATER
from ..weather import WeatherConditions
from . import physics_utils, pvt

__all__ = (
    "glass_temperature_gradient",
    "pv_temperature_gradient",
    "collector_temperature_gradient",
    "bulk_water_temperature_gradient",
)


def glass_temperature_gradient(
    collector_temperature: float,
    glass_temperature: float,
    pv_temperature: float,
    pvt_panel: pvt.PVT,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the temperature gradient of the glass layer of the PVT panel.

    :param colelctor_temperature:
        The temperature of the collector layer of the panel, measured in Kelvin.

    :param glass_temperature:
        The temperature of the glass layer of the panel, measured in Kelvin.

    :param pv_temperature:
        The temperature of the PV layer of the panel, measured in Kelvin.

    :param pvt_panel:
        An instance representing the PVT panel.

    :param weather_conditions:
        The current weather conditions.

    :return:
        The temperature gradient of the glass layer of the panel, measured in Kelvin per
        second.

    """

    return (
        # PV to Glass conductive heat input
        physics_utils.conductive_heat_transfer_with_gap(
            air_gap_thickness=pvt_panel.air_gap_thickness,
            contact_area=pvt_panel.pv.area,
            destination_temperature=glass_temperature,
            source_temperature=pv_temperature,
        )  # [W]
        # PV to Glass radiative heat input
        + physics_utils.radiative_heat_transfer(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=glass_temperature,
            radiating_to_sky=False,
            radiative_contact_area=pvt_panel.pv.area,
            source_emissivity=pvt_panel.pv.emissivity,
            source_temperature=pv_temperature,
        )  # [W]
        # Collector to Glass conductive heat input
        + physics_utils.conductive_heat_transfer_with_gap(
            air_gap_thickness=pvt_panel.air_gap_thickness,
            contact_area=(1 - pvt_panel.portion_covered) * pvt_panel.area,
            destination_temperature=glass_temperature,
            source_temperature=collector_temperature,
        )  # [W]
        # Collector to Glass radiative heat input
        + physics_utils.radiative_heat_transfer(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=glass_temperature,
            radiating_to_sky=False,
            radiative_contact_area=(1 - pvt_panel.portion_covered) * pvt_panel.area,
            source_emissivity=pvt_panel.collector.emissivity,
            source_temperature=collector_temperature,
        )  # [W]
        # Wind to Glass heat input
        + physics_utils.wind_heat_transfer(
            contact_area=pvt_panel.area,
            destination_temperature=glass_temperature,
            source_temperature=weather_conditions.ambient_temperature,
            wind_heat_transfer_coefficient=weather_conditions.wind_heat_transfer_coefficient,
        )
        # Sky to Glass heat input
        - physics_utils.radiative_heat_transfer(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            radiative_contact_area=pvt_panel.area,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=glass_temperature,
        )  # [W]
    ) / (
        pvt_panel.glass.mass * pvt_panel.glass.heat_capacity  # [J/K]
    )


def pv_temperature_gradient(
    collector_temperature: float,
    glass_temperature: float,
    pv_temperature: float,
    pvt_panel: pvt.PVT,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the temperature gradient of the pv layer of the PVT panel.

    :param colelctor_temperature:
        The temperature of the collector layer of the panel, measured in Kelvin.

    :param glass_temperature:
        The temperature of the glass layer of the panel, measured in Kelvin.

    :param pv_temperature:
        The temperature of the PV layer of the panel, measured in Kelvin.

    :param pvt_panel:
        An instance representing the PVT panel.

    :param weather_conditions:
        The current weather conditions.

    :return:
        The temperature gradient of the pv layer of the panel, measured in Kelvin per
        second.

    """

    return (
        # Solar heat input
        physics_utils.solar_heat_input(
            pvt_panel.pv_area,
            weather_conditions.solar_energy_input,
            physics_utils.transmissivity_absorptivity_product(
                diffuse_reflection_coefficient=pvt_panel.glass.diffuse_reflection_coefficient,  # pylint: disable=line-too-long
                glass_transmissivity=pvt_panel.glass.transmissivity,
                layer_absorptivity=pvt_panel.pv.absorptivity,
            ),
        )  # [W]
        # Collector to PV heat input
        + physics_utils.conductive_heat_transfer_no_gap(
            contact_area=pvt_panel.pv.area,
            destination_temperature=pv_temperature,
            source_temperature=collector_temperature,
            thermal_conductance=pvt_panel.pv_to_collector_thermal_conductance,
        )  # [W]
        # Glass to PV conductive heat input
        + physics_utils.conductive_heat_transfer_with_gap(
            air_gap_thickness=pvt_panel.air_gap_thickness,
            contact_area=pvt_panel.pv.area,
            destination_temperature=pv_temperature,
            source_temperature=glass_temperature,
        )  # [W]
        # Glass to PV radiative heat input
        + physics_utils.radiative_heat_transfer(
            destination_emissivity=pvt_panel.pv.emissivity,
            destination_temperature=pv_temperature,
            radiating_to_sky=False,
            radiative_contact_area=pvt_panel.pv.area,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=glass_temperature,
        )  # [W]
    ) / (
        pvt_panel.pv.mass * pvt_panel.pv.heat_capacity
    )  # [J/K]


def collector_temperature_gradient(
    bulk_water_temperature: float,
    collector_temperature: float,
    glass_temperature: float,
    pv_temperature: float,
    pvt_panel: pvt.PVT,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the temperature gradient of the collector layer of the PVT panel.

    :param bulk_water_temperature:
        The temperature of the bulk water, mesaured in Kelvin.

    :param colelctor_temperature:
        The temperature of the collector layer of the panel, measured in Kelvin.

    :param glass_temperature:
        The temperature of the glass layer of the panel, measured in Kelvin.

    :param pv_temperature:
        The temperature of the PV layer of the panel, measured in Kelvin.

    :param pvt_panel:
        An instance representing the PVT panel.

    :param weather_conditions:
        The current weather conditions.

    :return:
        The temperature gradient of the collector layer of the panel, measured in Kelvin
        per second.

    """

    return (
        # Solar heat input
        physics_utils.solar_heat_input(
            (1 - pvt_panel.portion_covered) * pvt_panel.area,
            weather_conditions.solar_energy_input,
            physics_utils.transmissivity_absorptivity_product(
                diffuse_reflection_coefficient=pvt_panel.glass.diffuse_reflection_coefficient,  # pylint: disable=line-too-long
                glass_transmissivity=pvt_panel.glass.transmissivity,
                layer_absorptivity=pvt_panel.collector.absorptivity,
            ),
        )  # [W]
        # PV to Collector heat input
        + physics_utils.conductive_heat_transfer_no_gap(
            contact_area=pvt_panel.pv.area,
            destination_temperature=collector_temperature,
            source_temperature=pv_temperature,
            thermal_conductance=pvt_panel.pv_to_collector_thermal_conductance,
        )  # [W]
        # Glass to Collector conductive heat input
        + physics_utils.conductive_heat_transfer_with_gap(
            air_gap_thickness=pvt_panel.air_gap_thickness,
            contact_area=(1 - pvt_panel.portion_covered) * pvt_panel.area,
            destination_temperature=collector_temperature,
            source_temperature=glass_temperature,
        )  # [W]
        # Glass to Collector radiative heat input
        + physics_utils.radiative_heat_transfer(
            destination_emissivity=pvt_panel.collector.emissivity,
            destination_temperature=collector_temperature,
            radiating_to_sky=False,
            radiative_contact_area=(1 - pvt_panel.portion_covered) * pvt_panel.area,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=glass_temperature,
        )  # [W]
        # Bulk Water to Collector convective heat input
        - physics_utils.convective_heat_transfer_to_fluid(
            contact_area=pvt_panel.collector.htf_surface_area,
            convective_heat_transfer_coefficient=pvt_panel.collector.convective_heat_transfer_coefficient_of_water,  # pylint: disable=line-too-long
            fluid_temperature=bulk_water_temperature,
            wall_temperature=collector_temperature,
        )  # [W]
    ) / (
        pvt_panel.collector.mass * pvt_panel.collector.heat_capacity
    )  # [J/K]


def bulk_water_temperature_gradient(
    bulk_water_temperature: float, collector_temperature: float, pvt_panel: float
) -> float:
    """
    Computes the temperature gradient of the bulk water.

    :param bulk_water_temperature:
        The temperature of the bulk water, mesaured in Kelvin.

    :param colelctor_temperature:
        The temperature of the collector layer of the panel, measured in Kelvin.

    :param pvt_panel:
        An instance representing the PVT panel.

    :return:
        The temperature gradient of the bulk water, measured in Kelvin.

    """

    return (
        physics_utils.convective_heat_transfer_to_fluid(
            contact_area=pvt_panel.collector.htf_surface_area,
            convective_heat_transfer_coefficient=pvt_panel.collector.convective_heat_transfer_coefficient_of_water,  # pylint: disable=line-too-long
            fluid_temperature=bulk_water_temperature,
            wall_temperature=collector_temperature,
        )  # [W]
    ) / (
        pvt_panel.collector.htf_volume * DENSITY_OF_WATER
    )  # [J/K]
