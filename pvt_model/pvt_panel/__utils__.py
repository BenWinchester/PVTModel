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

from typing import Optional

from ..__utils__ import (
    LayerParameters,
    OpticalLayerParameters,
    STEFAN_BOLTZMAN_CONSTANT,
    THERMAL_CONDUCTIVITY_OF_AIR,
)

__all__ = (
    "conductive_heat_transfer_no_gap",
    "conductive_heat_transfer_with_gap",
    "Layer",
    "OpticalLayer",
    "radiative_heat_transfer",
    "solar_heat_input",
)


class Layer:
    """
    Represents a layer within the PV-T panel.

    .. attribute:: area
        The area of the layer, measured in meters squared.

    .. attribute:: temperature
        The temperature of the layer, measured in Kelvin.

    .. attribute:: thickenss
        The thickness (depth) of the layer, measured in meters.

    """

    # Private attributes:
    #
    # .. attribute:: _heat_capacity
    #   The heat capacity of the layer, measured in Joules per kilogram Kelvin.
    #
    # .. attribute:: _mass
    #   The mass of the layer in kilograms.
    #

    def __init__(self, layer_params: LayerParameters) -> None:
        """
        Instantiate an instance of the layer class.

        """

        self._heat_capacity = layer_params.heat_capacity  # [J/kg*K]
        self._mass = layer_params.mass  # [kg]
        self.area = layer_params.area
        self.temperature = (
            layer_params.temperature if layer_params.temperature is not None else 293
        )
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
                optical_params.mass,  # [kg]
                optical_params.heat_capacity,
                optical_params.area,
                optical_params.thickness,
                optical_params.temperature,
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
            f"_heat_capacity: {self._heat_capacity}, "
            f"_mass: {self._mass}, "
            f"absorptivity: {self.absorptivity}, "
            f"area: {self.area}, "
            f"emissivitiy: {self.emissivity}, "
            f"temperature: {self.temperature}, "
            f"thickness: {self.thickness}, "
            f"transmissivity: {self.transmissivity}"
            ")"
        )

    def _layer_to_sky_radiative_transfer(
        self, fraction_emitting: float, sky_temperature: float
    ) -> float:
        """
        Calculates the heat loss to the sky radiatively from the layer in Watts.

        :param fraction_emitting:
            The fraction of the layer which is emitting to the sky.

        :param sky_temperature:
            The radiative temperature of the sky, measured in Kelvin.

        :return:
            The heat transfer, in Watts, radiatively to the sky.

        """

        return (
            self.emissivity
            * STEFAN_BOLTZMAN_CONSTANT
            * self.area
            * fraction_emitting
            * (self.temperature ** 4 - sky_temperature ** 4)
        )

    def _layer_to_air_convective_transfer(
        self,
        ambient_temperature: float,
        fraction_emitting: float,
        wind_heat_transfer_coefficient: float,
    ) -> float:
        """
        Calculates the heat loss to the surrounding air by conduction and convection in
        Watts.

        :param ambient_temperature:
            The ambient temperature of the air, measured in Kelvin.

        :param fraction_emitting:
            The fraction of the layer which is emitting to the air.

        :param wind_heat_transfer_coefficient:
            The convective heat transfer coefficient due to the wind, measured in W/K.

        :return:
            The heat transfer, in Watts, conductively to the surrounding air.

        """

        return (
            wind_heat_transfer_coefficient  # [W/m^2*K]
            * self.area  # [m^2]
            * fraction_emitting
            * (self.temperature - ambient_temperature)  # [K]
        )


def conductive_heat_transfer_no_gap(
    *,
    contact_area: float,
    destination_temperature: float,
    source_temperature: float,
    thermal_conductance: float,
) -> float:
    """
    Computes the heat transfer between two layers that are in thermal contact.

    The value computed is positive if the heat transfer is from the source to the
    destination, as determined by the arguments, and negative if the flow of heat is
    the reverse of what is implied via the parameters.

    The value for the heat transfer is returned in Watts.

    :param contact_area:
        The area of contact between the two layers over which conduction can occur,
        measured in meters squared.

    :param destination_temperature:
        The temperature of the destination layer/material, measured in Kelvin.

    :param source_temperature:
        The temperature of the source layer/material, measured in Kelvin.

    :param thermal_conductance:
        The conductance, measured in Watts per meter squared Kelvin, between the two
        layers/materials.

    :return:
        The heat transfer, in Watts, from the PV layer to the collector layer.

    """

    return (
        thermal_conductance  # [W/m^2*K]
        * (source_temperature - destination_temperature)  # [K]
        * contact_area  # [m^2]
    )  # [W]


def conductive_heat_transfer_with_gap(
    air_gap_thickness: float,
    contact_area: float,
    destination_temperature: float,
    source_temperature: float,
) -> float:
    """
    Computes the conductive heat transfer between the two layers.

    The value computed is positive if the heat transfer is from the source to the
    destination, as determined by the arguments, and negative if the flow of heat is
    the reverse of what is implied via the parameters.

    The value for the heat transfer is returned in Watts.

    :param air_gap_thickness:
        The thickness of the air gap between the PV and glass layers.

    :param contact_area:
        The area of contact between the two layers over which conduction can occur,
        measured in meters squared.

    :param destination_temperature:
        The temperature of the destination layer/material, measured in Kelvin.

    :param source_temperature:
        The temperature of the source layer/material, measured in Kelvin.

    :return:
        The heat transfer, in Watts, from the source layer to the destination layer that
        takes place by conduction.

    """

    return (
        THERMAL_CONDUCTIVITY_OF_AIR  # [W/m*K]
        * (source_temperature - destination_temperature)  # [K]
        * contact_area  # [m^2]
        / air_gap_thickness  # [m]
    )  # [W]


def radiative_heat_transfer(
    destination_emissivity: float,
    destination_temperature: float,
    radiative_contact_area: float,
    source_emissivity: float,
    source_temperature: float,
) -> float:
    """
    Computes the radiative heat transfer between two layers.

    The value computed is positive if the heat transfer is from the source to the
    destination, as determined by the arguments, and negative if the flow of heat is
    the reverse of what is implied via the parameters.

    The value for the heat transfer is returned in Watts.

    :param destination_emissivity:
        The emissivity of the layer that is receiving the radiation, defined between 0
        and 1.

    :param destination_temperature:
        The temperature of the destination layer/material, measured in Kelvin.

    :param radiative_contact_area:
        The area of contact between the two layers over which radiation can occur,
        measured in meters squared.

    :param source_temperature:
        The temperature of the source layer/material, measured in Kelvin.

    :param source_emissivity:
        The emissivity of the layer that is radiating, defined between 0 and 1.

    :return:
        The heat transfer, in Watts, from the PV layer to the glass layer that takes
        place by radiative transfer.

    """

    return (
        STEFAN_BOLTZMAN_CONSTANT  # [W/m^2*K^4]
        * radiative_contact_area  # [m^2]
        * (source_temperature ** 4 - destination_temperature ** 4)  # [K^4]
    ) / ((1 / source_emissivity) + (1 / destination_emissivity) - 1)


def solar_heat_input(
    absorptivity: float,
    area: float,
    solar_energy_input: float,
    transmissivity: float,
    electrical_efficiency: Optional[float] = None,
) -> float:
    """
    Determines the heat input due to solar irradiance, measured in Joules per time step.

    The nature of the panel being only partially-covered with photovoltaic cells means
    that, for some of the panel, a fraction of the light is removed as electrical
    energy (the PV-covered section), and that, for some of the panel, all of the
    incident light that is absorbed by the panel is converted into heat.

    :param absorptivity:
        The absorptivity of the layer: this is a dimensionless number defined between
        0 (no light is absorbed) and 1 (all incident light is absorbed).

    :param area:
        The area of the layer, measured in meters squared.

    :param solar_energy_input:
        The solar energy input, normal to the panel, measured in Energy per meter
        squared per resolution time interval.

    :param transmissivity:
        The transmissivity of the layer: this is a dimensionless number defined between
        0 (no light is transmitted through the PV layer) and 1 (all incident light is
        transmitted).

    :param electrical_efficiency:
        The electrical conversion efficiency of the layer, defined between 0 and 1.

    :return:
        The solar heating input, measured in Watts.

    """

    # If the layer is not electrical, compute the input as a thermal-only layer.
    if electrical_efficiency is None:
        return (
            (transmissivity * absorptivity)
            * solar_energy_input  # [J/time_step*m^2]
            * area  # [m^2]
        )

    return (
        (transmissivity * absorptivity)
        * solar_energy_input  # [J/time_step*m^2]
        * area  # [m^2]
        * (1 - electrical_efficiency)
    )
