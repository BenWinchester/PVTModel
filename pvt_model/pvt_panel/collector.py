#!/usr/bin/python3.7
########################################################################################
# pvt_panel/collector.py - Represents a collector within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The collector module for the PV-T model.

This module represents a thermal collector within a PV-T panel.

"""

import math
import pdb

from typing import Optional, Tuple

from . import back_plate

from ..__utils__ import (
    CollectorParameters,
    DENSITY_OF_WATER,
    HEAT_CAPACITY_OF_WATER,
    get_logger,
    LOGGER_NAME,
    OpticalLayerParameters,
    ProgrammerJudgementFault,
    WeatherConditions,
)
from .__utils__ import (
    conductive_heat_transfer_no_gap,
    conductive_heat_transfer_with_gap,
    convective_heat_transfer_to_fluid,
    OpticalLayer,
    radiative_heat_transfer,
    wind_heat_transfer,
)

__all__ = ("Collector",)

# Get the logger for the run.
logger = get_logger(LOGGER_NAME)


class Collector(OpticalLayer):
    """
    Represents the thermal collector (lower) layer of the PV-T panel.

    .. attribute:: htf_heat_capacity
        The heat capacity of the heat-transfer fluid passing through the collector,
        measured in Joules per kilogram Kelvin.

    .. attribute:: mass_flow_rate
        The mass flow rate of heat-transfer fluid through the collector, measured in
        kilograms per second.

    .. attribute:: output_water_temperature
        The temperature of the water outputted by the layer, measured in Kelvin.

    .. attribute:: pump_power
        The power consumed by the water pump, measured in Watts.

    """

    # Pirvate Attributes:
    #
    # .. attribute:: _mass_flow_rate
    #   The mass flow rate of heat-trasnfer fluid through the collector, measured in
    #   Litres per hour.
    #

    def __init__(self, collector_params: CollectorParameters) -> None:
        """
        Instantiate a collector layer.

        :param collector_params:
            The parameters needed to instantiate the collector.

        """

        super().__init__(
            OpticalLayerParameters(
                collector_params.mass,  # [kg]
                collector_params.heat_capacity,
                collector_params.area,
                collector_params.thickness,
                collector_params.temperature,
                collector_params.transmissivity,
                collector_params.absorptivity,
                collector_params.emissivity,
            )
        )

        self._length = collector_params.length
        self._mass_flow_rate = collector_params.mass_flow_rate
        self._number_of_pipes = collector_params.number_of_pipes
        self._pipe_diameter = collector_params.pipe_diameter
        self.bulk_water_temperature = collector_params.bulk_water_temperature
        self.htf_heat_capacity = collector_params.htf_heat_capacity
        self.output_water_temperature = collector_params.output_water_temperature
        self.pump_power = collector_params.pump_power

    def __repr__(self) -> str:
        """
        Returns a nice representation of the layer.

        :return:
            A `str` giving a nice representation of the layer.

        """

        return (
            "Collector("
            f"_heat_capacity: {self._heat_capacity}J/kg*K, "
            f"_mass: {self._mass}kg, "
            f"area: {self.area}m^2, "
            f"bulk_water_temperature: {self.bulk_water_temperature}, "
            f"htf_heat_capacity: {self.htf_heat_capacity}J/kg*K)"
            f"mass_flow_rate: {self.mass_flow_rate}kg/s, "
            f"output_temperature: {self.output_water_temperature}K, "
            f"temperature: {self.temperature}K, "
            f"thickness: {self.thickness}m, "
            ")"
        )

    @property
    def convective_heat_transfer_coefficient_of_water(self) -> float:
        """
        Returns the convective heat transfer coefficient of water, measured in W/m^2*K.

        :return:
            The convective heat transfer coefficient of water, calculated from the
            Nusselt number for the flow, the conductivity of water, and the pipe
            diameter.

        """

        # @@@ Maria here used a value of 259, irrespective of these properties.
        # @@@ For temporary consistency, this value is used.

        return 259

        # return NUSSELT_NUMBER * THERMAL_CONDUCTIVITY_OF_WATER / self._pipe_diameter

    @property
    def htf_surface_area(self) -> float:
        """
        Returns the contact area between the HTF and the collector, measured in m^2.

        :return:
            The contact surface area, between the collector (i.e., the pipes) and the
            HTF passing through the pipes.
            A single pass is assumed, with multiple pipes increasing the area, rather
            than the length, of the collector.

        """

        return (
            self._number_of_pipes  # [pipes]
            * math.pi
            * self._pipe_diameter  # [m]
            * self._length  # [m]
        )

    @property
    def htf_volume(self) -> float:
        """
        Returns the volume of HTF that can be held within the collector, measured in m^3

        :return:
            The volume of the HTF within the collector, measured in meters cubed.

        """

        return (
            self._number_of_pipes  # [pipes]
            * math.pi
            * (self._pipe_diameter) ** 2  # [m^2]
            * self._length
        )

    @property
    def mass_flow_rate(self) -> float:
        """
        Return the mass-flow rate in kilograms per second.

        :return:
            d/dt(M) in kg/s

        """

        return self._mass_flow_rate / (3600)  # [kg/s]

    def update(
        self,
        *,
        air_gap_thickness: float,
        back_plate_instance: back_plate.BackPlate,
        collector_heat_input: float,
        glass_emissivity: Optional[float],
        glass_layer_included: bool,
        glass_temperature: Optional[float],
        input_water_temperature: float,
        internal_resolution: float,
        portion_covered: float,
        weather_conditions: WeatherConditions,
    ) -> Tuple[float, Optional[float], float, float, float]:
        """
        Update the internal properties of the PV layer based on external factors.

        :param air_gap_thickness:
            The thickness, measured in meters, of the air gap between the glass layer
            and the rest of the panel. This parameter is only needed when there is no PV
            layer but the thermal collector is glazed. In this case, the collector layer
            experiences a radiative heat loss to the glass layer.

        :param back_plate_instance:
            The back plate of the PV-T panel, through which heat is lost.

        :param collector_heat_input:
            Heat inputted to the collector layer, measured in Joules.

        :param glass_emissivity:
            The emissivity of the glass layer, if present. If no glass layer is present,
            then this is None.

        :param glass_layer_included:
            Whether there is a glass layer present, radiating to the sky, or whether the
            collector layer (or PV layer) is exposed directly to the sky.

        :param glass_temperature:
            The temperature of the glass layer, if present, measured in Kelvin. If there
            is no glass layer present, then this is None.

        :param input_water_temperature:
            The temperature of the input water flow to the collector, measured in
            Kelvin, at the current time step.

        :param internal_resolution:
            The resolution at which the simulation is being run, measured in seconds.

        :param portion_covered:
            The portion of the PV-T panel which is covered with PV.

        :param weather_conditions:
            The current weather conditions.

        :return:
            A `tuple` containing:
            - the heat loss through the back plate, measured in Joules;
            - the heat transferred to the HTF, measured in Joules;
            - the output water temperature from the collector;
            - the heat transferred to the glass layer, measured in Joules;
            - the upward heat loss, measured in Joules.

        """

        if self.temperature > 1000:
            logger.debug(
                "The temperature of the thermal collector is over 1000K. "
                "Importing the PDB debugger.\nCollector profile: %s",
                self,
            )
            pdb.set_trace(
                header="Thermal collector melting - temperature greater than 1000K."
            )

        # From the excess heat, compute what is not lost to the environment, and, from
        # there, what is transferred to the HTF.
        back_plate_heat_loss = (
            conductive_heat_transfer_no_gap(
                contact_area=self.area,
                destination_temperature=weather_conditions.ambient_temperature,
                source_temperature=back_plate_instance.temperature,
                thermal_conductance=back_plate_instance.conductance,
            )
            * internal_resolution
        )  # [seconds]  # [J]

        # If there are no glass or PV layers, then we lose heat from the collector
        # layer directly.
        if not glass_layer_included and portion_covered != 1:
            upward_heat_losses: float = (
                wind_heat_transfer(
                    contact_area=self.area * (1 - portion_covered),
                    destination_temperature=weather_conditions.ambient_temperature,
                    source_temperature=self.temperature,
                    wind_heat_transfer_coefficient=weather_conditions.wind_heat_transfer_coefficient,  # pylint: disable=line-too-long
                )  # [W]
                + radiative_heat_transfer(
                    destination_temperature=weather_conditions.sky_temperature,
                    radiating_to_sky=True,
                    radiative_contact_area=self.area * (1 - portion_covered),
                    source_emissivity=self.emissivity,
                    source_temperature=self.temperature,
                )  # [W]
                * internal_resolution
            )
        # If there is a glass layer, and a PV layer that does not fully cover the panel,
        # then we need to compute the energy transferred to the glass layer.
        elif glass_layer_included and portion_covered != 1:
            if glass_temperature is None or glass_emissivity is None:
                raise ProgrammerJudgementFault(
                    "The system attempted to compute a radiative and/or conductive "
                    "transfer to a non-existant glass layer."
                )
            upward_heat_losses = (
                radiative_heat_transfer(
                    destination_emissivity=glass_emissivity,
                    destination_temperature=glass_temperature,
                    radiative_contact_area=self.area * (1 - portion_covered),
                    source_emissivity=self.emissivity,
                    source_temperature=self.temperature,
                )  # [W]
                + conductive_heat_transfer_with_gap(
                    air_gap_thickness=air_gap_thickness,
                    destination_temperature=glass_temperature,
                    contact_area=self.area * (1 - portion_covered),
                    source_temperature=self.temperature,
                )  # [W]
            ) * internal_resolution  # [J]
        # Otherwise, if the collector is completely covered by a PV layer, then there
        # are no upward heat losses as these are encapsulated in the PV layer heat
        # transfer variable.
        else:
            upward_heat_losses = 0  # [J]

        # @@@
        # Check: The bulk-water temperature is computed via an average, see Hil '21 p.31
        htf_mass_affected = (
            DENSITY_OF_WATER * self.htf_volume
            + self.mass_flow_rate * internal_resolution
        )  # [kg]

        self.bulk_water_temperature = (
            DENSITY_OF_WATER * self.htf_volume * self.bulk_water_temperature  # [kg*K]
            + self.mass_flow_rate
            * internal_resolution
            * input_water_temperature  # [kg*K]
        ) / htf_mass_affected  # [kg]  # [K]

        # Compute the heat flow to the bulk water and the output water temperature.
        bulk_water_heat_gain = (
            convective_heat_transfer_to_fluid(
                contact_area=self.htf_surface_area,
                convective_heat_transfer_coefficient=self.convective_heat_transfer_coefficient_of_water,  # pylint: disable=line-too-long
                fluid_temperature=self.bulk_water_temperature,
                wall_temperature=self.temperature,
            )
            * internal_resolution
        )  # [J]
        self.bulk_water_temperature += bulk_water_heat_gain / (  # [J[
            htf_mass_affected * self.htf_heat_capacity  # [kg] * [J/kg*K]
        )
        self.output_water_temperature = (
            2 * self.bulk_water_temperature - input_water_temperature
        )

        # This heat is now converted into Joules.
        net_heat_gain = collector_heat_input - (  # [J]
            back_plate_heat_loss  # [J]
            + upward_heat_losses  # [J]
            + bulk_water_heat_gain  # [J]
        )
        self.temperature += net_heat_gain / (
            (self._mass + back_plate_instance.mass) * self._heat_capacity
        )
        back_plate_instance.temperature = self.temperature

        # >>> If there is a glass layer present, return the heat flow to it.
        if glass_layer_included:
            return (
                back_plate_heat_loss,  # [J]
                bulk_water_heat_gain,  # [J]
                self.output_water_temperature,  # [K]
                upward_heat_losses,  # [J]
                upward_heat_losses,  # [J]
            )
        # <<< Otherwise, return None
        return (
            back_plate_heat_loss,  # [J]
            bulk_water_heat_gain,  # [J]
            self.output_water_temperature,  # [K]
            None,
            upward_heat_losses,  # [J]
        )