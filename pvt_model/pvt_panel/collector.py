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
    get_logger,
    NUSSELT_NUMBER,
    LOGGER_NAME,
    OpticalLayerParameters,
    ProgrammerJudgementFault,
    THERMAL_CONDUCTIVITY_OF_WATER,
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

    .. attribute:: htfheat_capacity
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
        self.htfheat_capacity = collector_params.htfheat_capacity
        self.output_water_temperature = collector_params.output_water_temperature

    def __repr__(self) -> str:
        """
        Returns a nice representation of the layer.

        :return:
            A `str` giving a nice representation of the layer.

        """

        return (
            "Collector("
            f"heat_capacity: {self.heat_capacity}J/kg*K, "
            f"_mass: {self._mass}kg, "
            f"area: {self.area}m^2, "
            f"bulk_water_temperature: {self.bulk_water_temperature}, "
            f"htfheat_capacity: {self.htfheat_capacity}J/kg*K)"
            f"mass_flow_rate: {self.mass_flow_rate}kg/s, "
            f"output_temperature: {self.output_water_temperature}K, "
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

        # return 259

        convective_heat_transfer_coefficient = (
            NUSSELT_NUMBER * THERMAL_CONDUCTIVITY_OF_WATER / self._pipe_diameter
        )
        return convective_heat_transfer_coefficient

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
    ) -> Tuple[float, float, float, Optional[float], float]:
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
            Heat inputted to the collector layer, measured in Watts.

        :param glass_emissivity:
            The emissivity of the glass layer, if present. If no glass layer is present,
            then this is None.

        :param glass_layer_included:
            Whether there is a glass layer present, radiating to the sky, or whether the
            collector layer (or PV layer) is exposed directly to the sky.

        :param glass_temperature:
            The temperature of the glass layer, if present, measured in Kelvin. If there
            is no glass layer present, then this is None.

        :param htf_pump_state:
            Whether the HTF pump is on (True) or off (False).

        :param input_water_temperature:
            The temperature of the input water flow to the collector, measured in
            Kelvin, at the current time step.

        :param internal_resolution:
            The internal resolution of the model, measured in seconds.

        :param portion_covered:
            The portion of the PV-T panel which is covered with PV.

        :param weather_conditions:
            The current weather conditions.

        :return:
            A `tuple` containing:
            - the heat loss through the back plate, measured in Watts;
            - the heat transferred to the HTF, measured in Watts;
            - the output water temperature from the collector;
            - the heat transferred to the glass layer, measured in Watts;
            - the upward heat loss, measured in Watts.

        :raises: ProgrammerJudgementFault
            A :class:`..__utils__.ProgrammerJudgementFault` is raised if an attempt is
            made to access glass-laye rproperties that weren't supplied to the function.

        """

        # From the excess heat, compute what is not lost to the environment, and, from
        # there, what is transferred to the HTF.
        back_plate_heat_loss = conductive_heat_transfer_no_gap(
            contact_area=self.area,
            destination_temperature=weather_conditions.ambient_temperature,
            source_temperature=back_plate_instance.temperature,
            thermal_conductance=back_plate_instance.conductance,
        )  # [W]

        # If there are no glass or PV layers, then we lose heat from the collector
        # layer directly.
        if not glass_layer_included and portion_covered != 1:
            upward_heat_losses: float = wind_heat_transfer(
                contact_area=self.area * (1 - portion_covered),
                destination_temperature=weather_conditions.ambient_temperature,
                source_temperature=self.temperature,
                wind_heat_transfer_coefficient=weather_conditions.wind_heat_transfer_coefficient,  # pylint: disable=line-too-long
            ) + radiative_heat_transfer(  # [W]
                destination_temperature=weather_conditions.sky_temperature,
                radiating_to_sky=True,
                radiative_contact_area=self.area * (1 - portion_covered),
                source_emissivity=self.emissivity,
                source_temperature=self.temperature,
            )  # [W]
        # If there is a glass layer, and a PV layer that does not fully cover the panel,
        # then we need to compute the energy transferred to the glass layer.
        elif glass_layer_included and portion_covered != 1:
            if glass_temperature is None or glass_emissivity is None:
                raise ProgrammerJudgementFault(
                    "The system attempted to compute a radiative and/or conductive "
                    "transfer to a non-existant glass layer."
                )
            upward_heat_losses = radiative_heat_transfer(
                destination_emissivity=glass_emissivity,
                destination_temperature=glass_temperature,
                radiative_contact_area=self.area * (1 - portion_covered),
                source_emissivity=self.emissivity,
                source_temperature=self.temperature,
            ) + conductive_heat_transfer_with_gap(  # [W]
                air_gap_thickness=air_gap_thickness,
                destination_temperature=glass_temperature,
                contact_area=self.area * (1 - portion_covered),
                source_temperature=self.temperature,
            )  # [W]
        # Otherwise, if the collector is completely covered by a PV layer, then there
        # are no upward heat losses as these are encapsulated in the PV layer heat
        # transfer variable.
        else:
            upward_heat_losses = 0  # [W]

        # Check: The bulk-water temperature is computed via an average, see Hil '21 p.31
        # * Equation 10: Compute the heat transfer to the bulk water
        bulk_water_heat_gain = convective_heat_transfer_to_fluid(
            contact_area=self.htf_surface_area,
            convective_heat_transfer_coefficient=self.convective_heat_transfer_coefficient_of_water,  # pylint: disable=line-too-long
            fluid_temperature=self.bulk_water_temperature,
            wall_temperature=self.temperature,
        )  # [W]

        # * Compute the temperature rise of the bulk water.
        bulk_water_temperature_gain = (
            bulk_water_heat_gain  # [W]
            * 0.5  # @@@ MAGIC FACTOR!!!
            * internal_resolution  # [s]
            / (  # [W]
                self.htf_volume  # [m^3]
                * DENSITY_OF_WATER  # [kg/m^3]
                * self.htfheat_capacity  # [J/kg*K]
            )
        )

        self.bulk_water_temperature += bulk_water_temperature_gain

        # * Compute the output water temperature
        self.output_water_temperature = (
            self.bulk_water_temperature * 2 - input_water_temperature
        )

        # The net heat is computed.
        net_heat_gain = collector_heat_input - (  # [W]
            back_plate_heat_loss  # [W]
            + upward_heat_losses  # [W]
            + bulk_water_heat_gain  # [W]
        )

        # This heat is absorbed by the thermally-coupled collector-back-plate system.
        self.temperature += (
            net_heat_gain  # [W]
            * internal_resolution  # [s]
            * 0.5  # @@@ MAGIC FACTOR!!!
            / (
                self._mass * self.heat_capacity  # [kg]  # [J/kg*K]
                + back_plate_instance.mass  # [kg]
                * back_plate_instance.heat_capacity  # [J/kg*K]
            )
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
