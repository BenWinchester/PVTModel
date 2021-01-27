#!/usr/bin/python3.7
########################################################################################
# pvt_panel/pv.py - Represents a collector within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The collector module for the PV-T model.

This module represents a thermal collector within a PV-T panel.

"""

from typing import Optional, Tuple

from ..__utils__ import OpticalLayerParameters, PVParameters, WeatherConditions
from .__utils__ import (
    conductive_heat_transfer_no_gap,
    conductive_heat_transfer_with_gap,
    OpticalLayer,
    radiative_heat_transfer,
    solar_heat_input,
)

__all__ = ("PV",)


class PV(OpticalLayer):
    """
    Represents the photovoltaic (middle) layer of the PV-T panel.

    """

    # Private attributes:
    #
    # .. attribute:: _reference_efficiency
    #   The efficiency of the PV layer at the reference temperature. Thie value varies
    #   between 1 (corresponding to 100% efficiency), and 0 (corresponding to 0%
    #   efficiency)
    #
    # .. attribute:: _reference_temperature
    #   The referencee temperature, in Kelvin, at which the reference efficiency is
    #   defined.
    #
    # .. attribute:: _thermal_coefficient
    #   The thermal coefficient for the efficiency of the panel.
    #

    def __init__(self, pv_params: PVParameters) -> None:
        """
        Instantiate a PV layer.

        :param pv_params:
            Parameters needed to instantiate the PV layer.

        """

        super().__init__(
            OpticalLayerParameters(
                pv_params.mass,
                pv_params.heat_capacity,
                pv_params.area,
                pv_params.thickness,
                pv_params.temperature,
                pv_params.transmissivity,
                pv_params.absorptivity,
                pv_params.emissivity,
            )
        )

        self._reference_efficiency = pv_params.reference_efficiency
        self._reference_temperature = pv_params.reference_temperature
        self._thermal_coefficient = pv_params.thermal_coefficient

    def __repr__(self) -> str:
        """
        Returns a nice representation of the layer.

        :return:
            A `str` giving a nice representation of the layer.

        """

        return (
            "PV("
            f"absorptivity: {self.absorptivity}, "
            f"_heat_capacity: {self._heat_capacity}J/kg*K, "
            f"_mass: {self._mass}kg, "
            f"_reference_efficiency: {self._reference_efficiency}, "
            f"_reference_temperature: {self._reference_temperature}K, "
            f"_thermal_coefficient: {self._thermal_coefficient}K^(-1), "
            f"_transmissicity: {self.transmissivity}, "
            f"area: {self.area}m^2, "
            f"emissivity: {self.emissivity}, "
            f"temperature: {self.temperature}K, "
            f"thickness: {self.thickness}m"
            ")"
        )

    @property
    def electrical_efficiency(self) -> float:
        """
        Returns the electrical efficiency of the PV panel based on its temperature.

        :return:
            A decimal giving the percentage efficiency of the PV panel between 0 (0%
            efficiency), and 1 (100% efficiency).

        """

        return self._reference_efficiency * (  # [unitless]
            1
            - self._thermal_coefficient  # [1/K]
            * (self.temperature - self._reference_temperature)  # [K]
        )

    def update(
        self,
        air_gap_thickness: float,
        collector_temperature: float,
        glass_emissivity: Optional[float],
        glass_temperature: Optional[float],
        glazed: bool,
        internal_resolution: float,
        pv_to_collector_thermal_conductance: float,
        solar_energy_input: float,
        weather_conditions: WeatherConditions,
    ) -> Tuple[float, Optional[float]]:
        """
        Update the internal properties of the PV layer based on external factors.

        :param air_gap_thickness:
            The thickness of the gap between the glass and PV layers, measured in
            meters.

        :param collector_temperature:
            The temperature of the collector layer, measured in Kelvin.

        :param glass_emissivity:
            The emissivity of the glass layer.

        :param glass_temperature:
            The temperature glass layer of the PV-T panel in Kelvin.

        :param glazed:
            Whether or not the panel is glazed, I.E., whether the panel has a glass
            layer or not.

        :param internal_resolution:
            The resolution of the model being run, measured in seconds.

        :param pv_to_collector_thermal_conductance:
            The thermal conductance between the PV and collector layers, measured in
            Watts per meter squared Kelvin.

        :param solar_energy_input:
            The solar irradiance, normal to the panel, measured in Joules per meter
            sqaured per time interval.

        :param weather_conditions:
            The current weather conditions, passed in as a :class:`WeatherConditions`
            instance.

        :return:
            The heat transferred to the collector and the glass layers respectively as a
            `Tuple`. Both these values are measured in Joules.
            If there is no glass layer present, then `None` is returned as the value of
            the heat transfered to the glass layer.

        """

        # Determine the excess heat that has been inputted into the panel during this
        # time step, measured in Joules.
        solar_heat_gain = solar_heat_input(
            self.absorptivity,
            self.area,
            solar_energy_input,
            self.transmissivity,
            self.electrical_efficiency,
        )  # [J] or [J/time_step]

        # >>> If the layer is glazed, compute radiative and conductive heat to the glass
        if glazed:
            radiative_loss_upwards = (
                radiative_heat_transfer(
                    destination_emissivity=glass_emissivity,
                    destination_temperature=glass_temperature,
                    radiative_contact_area=self.area,
                    source_emissivity=self.emissivity,
                    source_temperature=self.temperature,
                )  # [W]
                * internal_resolution  # [seconds]
            )  # [J]
            convective_loss_upwards = (
                conductive_heat_transfer_with_gap(
                    air_gap_thickness=air_gap_thickness,
                    contact_area=self.area,
                    destination_temperature=glass_temperature,
                    source_temperature=self.temperature,
                )  # [W]
                * internal_resolution  # [seconds]
            )  # [J]
        # <<< If the layer is unglazed, compute losses to the sky and air.
        else:
            radiative_loss_upwards = (
                self._layer_to_sky_radiative_transfer(
                    1,
                    weather_conditions.sky_temperature,
                )  # [W]
                * internal_resolution  # [seconds]
            )  # [J]

            convective_loss_upwards = (
                self._layer_to_air_convective_transfer(
                    weather_conditions.ambient_temperature,
                    1,
                    weather_conditions.wind_heat_transfer_coefficient,
                )  # [W]
                * internal_resolution  # [seconds]
            )  # [J]

        # Compute the downward losses from the PV layer.
        pv_to_collector = (
            conductive_heat_transfer_no_gap(
                contact_area=self.area,
                destination_temperature=collector_temperature,
                source_temperature=self.temperature,
                thermal_conductance=pv_to_collector_thermal_conductance,
            )  # [W]
            * internal_resolution  # [seconds]
        )  # [J]

        # Compute the overall heat lost.
        heat_lost = (
            radiative_loss_upwards + convective_loss_upwards + pv_to_collector
        )  # [J]

        # Use this to compute the rise in temperature of the PV layer and set the
        # temperature appropriately.
        self.temperature += (solar_heat_gain - heat_lost) / (  # [J]
            self._mass * self._heat_capacity  # [kg]  # [J/kg*K]
        )  # [K]

        # Return the heat transfered to the glass and collector layers.
        # >>> If the collector is glazed, return the heat transfered to the glass layer.
        if glazed:
            return (
                pv_to_collector,  # [J]
                (
                    (radiative_loss_upwards + convective_loss_upwards)  # [J] + [J]
                    / (internal_resolution)  # [seconds]
                ),  # [W]
            )
        # <<< Otherwise, return None.
        return (pv_to_collector, None)  # [J] [W]
