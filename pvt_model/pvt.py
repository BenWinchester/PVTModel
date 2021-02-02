#!/usr/bin/python3.7
########################################################################################
# pvt.py - Models a PVT panel and all contained components.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The PV-T module for the PV-T model.

This module models the PV-T panel and its components, including the glass cover, PV
panel, and thermal collector. The model works by incrimenting the parameters through at
increasing time steps, and the code here aims to emaulate this.

"""

import datetime
import math
import pdb

from typing import Optional, Tuple

from .__utils__ import (
    MissingParametersError,
    ProgrammerJudgementFault,
    LayerParameters,
    BackLayerParameters,
    CollectorParameters,
    OpticalLayerParameters,
    PVParameters,
    WeatherConditions,
    FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR,
    NUSSELT_NUMBER,
    STEFAN_BOLTZMAN_CONSTANT,
    THERMAL_CONDUCTIVITY_OF_AIR,
    THERMAL_CONDUCTIVITY_OF_WATER,
)

__all__ = ("PVT",)


# The minimum height in degrees that the sun must be above the horizon for light to
# strike the panel and be useful.
MINIMUM_SOLAR_DECLINATION = 5
# The maximum angle at which the panel can still be assumed to gain sunlight. This will
# limit both edge effects and prevent light reaching the panel from behind.
MAXIMUM_SOLAR_DIFF_ANGLE = 88


####################
# Helper Functions #
####################


def _solar_heat_input(
    solar_energy_input: float,
    pv_area: float,
    pv_efficiency: float,
    pv_absorptivity: float,
    pv_transmissivity: float,
) -> float:
    """
    Determines the heat input due to solar irradiance.

    :param solar_energy_input:
        The solar energy input, normal to the panel, measured in Energy per meter
        squared per resolution time interval.

    :param pv_area:
        The area of the PV layer in meters squared.

    :param pv_efficiency:
        The electrical conversion efficiency of the PV layer, defined between 0 and 1.

    :param pv_absorptivity:
        The absorptivity of the PV layer: this is a dimensionless number defined between
        0 (no light is absorbed) and 1 (all incident light is absorbed).

    :param pv_transmissivity:
        The transmissivity of the PV layer: this is a dimensionless number defined
        between 0 (no light is transmitted through the PV layer) and 1 (all incident
        light is transmitted).

    :return:
        The solar heating input, measured in Joules per time step.

    """

    return (
        (pv_transmissivity * pv_absorptivity)
        * solar_energy_input  # [J/time_step*m^2]
        * pv_area  # [m^2]
        * (1 - pv_efficiency)
    )


def _htf_heat_transfer(
    input_water_temperature: float,
    collector_temperature: float,
    collector_mass_flow_rate: float,
    collector_htf_heat_capacity: float,
) -> float:
    """
    Computes the heat transfer to the heat-transfer fluid.

    :param input_water_temperature:
        The input water temperature to the PV-T panel, measured in Kelvin.

    :param collector_temperature:
        The collector temperature, measured in Kelvin.

    :param collector_mass_flow_rate:
        The mass-flow rate of heat-transfer fluid through the collector, measured in
        kilograms per second.

    :param collector_htf_heat_capacity:
        The heat capacity of the heat-transfer fluid flowing through the collector,
        measured in Joules per kilogram Kelvin.

    :return:
        The heat transfer, in Watts, to the heat transfer fluid.

    """

    return (
        collector_mass_flow_rate  # [kg/s]
        * collector_htf_heat_capacity  # [J/kg*K]
        * (collector_temperature - input_water_temperature)  # [K]
    )  # [W]


def _pv_to_collector_conductive_transfer(
    pv_area: float,
    pv_to_collector_thermal_conductance: float,
    pv_temperature: float,
    collector_temperature: float,
) -> float:
    """
    Computes the heat transfer from the PV layer to the thermal collector.

    This value is returned in Watts.

    :param pv_area:
        The area of the panel.

    :param pv_to_collector_thermal_conductance:
        The conductance, measured in Watts per meter squared Kelvin.

    :param pv_temperature:
        The temperature of the PV collector, measured in Kelvin.

    :param collector_temperature:
        The temperature of the collector, measured in Kelvin.

    :return:
        The heat transfer, in Watts, from the PV layer to the collector layer.

    """

    return (
        pv_to_collector_thermal_conductance  # [W/m^2*K]
        * (pv_temperature - collector_temperature)  # [K]
        * pv_area  # [m^2]
    )  # [W]


# @@@ Unused function.
def _collector_heat_usage(
    ambient_temperature: float,
    input_water_temperature: float,
    collector_temperature: float,
    collector_mass_flow_rate: float,
    collector_htf_heat_capacity: float,
    back_plate_conductance: float,
    back_plate_area: float,
) -> float:
    """
    Computes the heat transfer from the PV layer to the thermal collector in Watts.

    This is computed by looking at the heat losses through the collector, both to the
    bulk-water flow and through the back plate of the PV-T panel.

    :param ambient_temperature:
        The ambient temperature in Kelvin.

    :param input_water_temperature:
        The input water temperature to the PV-T panel, measured in Kelvin.

    :param collector_temperature:
        The temperature of the collector layer, measured in Kelvin.

    :param collector_mass_flow_rate:
        The mass-flow rate of heat-transfer fluid through the collector, measured in
        kilograms per second.

    :param collector_htf_heat_capacity:
        The heat capacity of the heat-transfer fluid flowing through the collector,
        measured in Joules per kilogram Kelvin.

    :param back_plate_conductance:
        The conductance of the back plate of the panel, measured in Watts per meter
        squared Kelvin.

    :param back_plate_area:
        The area of the back plate, measured in meters squared.

    :return:
        The heat transfer, in Watts, from the PV layer to the thermal-collector
        layer.

    """

    back_plate_heat_loss = (
        back_plate_conductance  # [W/m^2*K]
        * (collector_temperature - ambient_temperature)  # [K]
        * back_plate_area  # [m^2]
    )  # [W]

    # It's necessary to divide by the model resolution to convert from the resolution
    # time-step to Watts.
    htf_heat_transfer = _htf_heat_transfer(
        input_water_temperature,
        collector_temperature,
        collector_mass_flow_rate,
        collector_htf_heat_capacity,
    )  # [W]

    return back_plate_heat_loss + htf_heat_transfer


def layer_to_glass_radiative_transfer(
    glass_temperature: float,
    glass_emissivity: float,
    pv_area: float,
    radiating_layer_temperature: float,
    radiating_layer_emissivity: float,
) -> float:
    """
    Computes the radiative heat transfer from the PV layer to the glass layer.

    :param glass_temperature:
        The temperature of the glass layer of the PV-T system.

    :param glass_emissivity:
        The emissivity of the glass layer of the PV-T system.

    :param pv_area:
        The area of the PV layer (and the entier PV-T system), measured in meters
        squared.

    :param radiating_layer_temperature:
        The temperature of the layer that is radiating to the glass layer, measured in
        Kelvin.

    :param radiating_layer_emissivity:
        The emissivity of the layer that is radiating to the glass layer, defined
        between 0 and 1.

    :return:
        The heat transfer, in Watts, from the PV layer to the glass layer that takes
        place by radiative transfer.

    """

    return (
        STEFAN_BOLTZMAN_CONSTANT  # [W/m^2*K^4]
        * pv_area  # [m^2]
        * (radiating_layer_temperature ** 4 - glass_temperature ** 4)  # [K^4]
    ) / (1 / radiating_layer_emissivity + 1 / glass_emissivity - 1)


def layer_to_glass_conductive_transfer(
    air_gap_thickness: float,
    glass_temperature: float,
    pv_area: float,
    pv_temperature: float,
) -> float:
    """
    Computes the conductive heat transfer between the PV layer and glass cover.
    The value returned is in Watts.

    :param air_gap_thickness:
        The thickness of the air gap between the PV and glass layers.

    :param glass_temperature:
        The temperature of the glass layer of the PV-T system.

    :param pv_area:
        The area of the PV layer (and hence the whole PV-T system), measured in meters
        squared.

    :param pv_temperature:
        The temperature of the PV layer, measured in Kelvin.

    :return:
        The heat transfer, in Watts, from the PV layer to the glass layer that takes
        place by conduction.

    """

    return (
        THERMAL_CONDUCTIVITY_OF_AIR  # [W/m*K]
        * (pv_temperature - glass_temperature)  # [K]
        * pv_area  # [m^2]
        / air_gap_thickness  # [m]
    )  # [W]


#####################
# PV-T Panel Layers #
#####################


class _Layer:
    """
    Represents a layer within the PV-T panel.

    .. attribute:: temperature
        The temperature of the layer, measured in Kelvin.

    .. attribute:: area
        The area of the layer, measured in meters squared.

    .. attribute:: thickenss
        The thickness (depth) of the layer, measured in meters.

    """

    # Private attributes:
    #
    # .. attribute:: _mass
    #   The mass of the layer in kilograms.
    #
    # .. attribute:: _heat_capacity
    #   The heat capacity of the layer, measured in Joules per kilogram Kelvin.
    #

    def __init__(self, layer_params: LayerParameters) -> None:
        """
        Instantiate an instance of the layer class.

        """

        self._mass = layer_params.mass  # [kg]
        self._heat_capacity = layer_params.heat_capacity  # [J/kg*K]
        self.temperature = (
            layer_params.temperature if layer_params.temperature is not None else 293
        )
        self.area = layer_params.area
        self.thickness = layer_params.thickness


class BackPlate(_Layer):
    """
    Represents the back-plate layer of the PV-T panel.

    .. attribute:: conductance
        The conducance, measured in Watts per meter squared Kelvin, of the back layer to
        the surroundings.

    """

    def __init__(self, back_params: BackLayerParameters) -> None:
        """
        Instantiate a back layer instance.

        :param back_params:
            The parameters needed to instantiate the back layer of the panel.

        """

        super().__init__(
            LayerParameters(
                back_params.mass,
                back_params.heat_capacity,
                back_params.area,
                back_params.thickness,
                back_params.temperature,
            )
        )

        self.conductivity = back_params.conductivity

    def __repr__(self) -> str:
        """REPR"""
        return (
            "BackPlate(mass: {}, heat_capacity: {}, area: {}, thickness: {}, ".format(
                self._mass,  # [J/kg*K]
                self._heat_capacity,  # [J/kg*K]
                self.area,
                self.thickness,
            )
            + "temp: {}, conductivity: {})".format(self.temperature, self.conductivity)
        )

    @property
    def conductance(self) -> float:
        """
        Returns the conductance of the back plate in Watts per meters squared Kelvin.

        :return:
            The conductance of the layer, measured in Watts per meter squared Kelvin.

        """

        return (
            self.thickness / self.conductivity  # [m] / [W/m*K]
            + 1 / FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR  # [W/m^2*K]^-1
        )


class _OpticalLayer(_Layer):
    """
    Represents a layer within the PV-T panel that has optical properties.

    .. attribute:: emissivity
        The emissivity of the layer; a dimensionless number between 0 (nothing is
        emitted by the layer) and 1 (the layer re-emits all incident light).

    """

    # Private Attributes:
    #
    # .. attribute:: _transmissivity
    #   The transmissivity of the layer: a dimensionless number between 0 (nothing is
    #   transmitted through the layer) and 1 (all light is transmitted).
    #
    # .. attribute:: _absorptivity
    #   The absorptivity of the layer: a dimensionless number between 0 (nothing is
    #   absorbed by the layer) and 1 (all light is absorbed).
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

        self._absorptivity = optical_params.absorptivity
        self._transmissivity = optical_params.transmissivity
        self.emissivity = optical_params.emissivity

    def __repr__(self) -> str:
        """REPR"""

        return (
            "OpticalLayer(mass: {}, heat_capacity: {}, area: {}, thickness: {}, ".format(
                self._mass,  # [kg]
                self._heat_capacity,  # [J/kg*K]
                self.area,
                self.thickness,
            )
            + "temp: {}, transmissivity: {}, absorptivity: {}, ".format(
                self.temperature, self._transmissivity, self._absorptivity
            )
            + "emissivitiy: {})".format(self.emissivity)
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

    # @@@ The unused-import flag is disabled as wind_speed will be included eventually
    def _layer_to_air_convective_transfer(
        self,
        ambient_temperature: float,
        fraction_emitting: float,
        wind_heat_transfer_coefficient: float,  # pylint: disable=unused-import
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


class Glass(_OpticalLayer):
    """
    Represents the glass (upper) layer of the PV-T panel.

    """

    def update(
        self,
        internal_resolution: float,
        air_gap_thickness: float,
        pv_area: float,
        weather_conditions: WeatherConditions,
        heat_input: float,
    ) -> None:
        """
        Update the internal properties of the PV layer based on external factors.

        :param internal_resolution:
            The resolution of the simulation currently being run, measured in seconds.

        :param air_gap_thickness:
            The thickness, in meters, of the air gap between the PV and glass layers.

        :param pv_area:
            The area of the PV panel, measured in meters squared.

        :param weather_conditions:
            The weather conditions at the current time step.

        :param heat_input:
            The heat inputted to the glass layer, measured in Watts.

        """

        # Set the temperature of this layer appropriately.
        # ! All in Watts now
        # @@@
        # @   Here, the glass layer just emits to the sky as if it covers the whole
        # @ panel. I.E., there is no scope for a partially-glazed panel. This variable
        # @ would need to be taken into account here.
        heat_losses = self._layer_to_air_convective_transfer(
            weather_conditions.ambient_temperature,
            fraction_emitting=1,
            wind_heat_transfer_coefficient=weather_conditions.wind_heat_transfer_coefficient,
        ) + self._layer_to_sky_radiative_transfer(
            fraction_emitting=1, sky_temperature=weather_conditions.sky_temperature
        )  # [W]

        # This heat input, in Watts, is supplied throughout the duration, and so does
        # not need to be multiplied by the resolution.
        self.temperature = self.temperature + (  # [K]
            heat_input - heat_losses
        ) * internal_resolution / (  # [W] * [seconds]
            self._mass * self._heat_capacity
        )  # [kg] * [J/kg*K]


class PV(_OpticalLayer):
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
        """REPR"""

        return (
            "PV(mass: {}, heat_capacity: {}, area: {}, thickness: {}, ".format(
                self._mass,  # [kg]
                self._heat_capacity,  # [J/kg*K]
                self.area,
                self.thickness,
            )
            + "temp: {}, transmissivity: {}, absorptivity: {}, ".format(
                self.temperature, self._transmissivity, self._absorptivity
            )
            + "emissivitiy: {}, _reference_efficiency: {}, ".format(
                self.emissivity, self._reference_efficiency
            )
            + "_reference_temp: {}, thermal_coefficient: {}, efficiency: {})".format(
                self._reference_temperature, self._thermal_coefficient, self.efficiency
            )
        )

    def update(
        self,
        glazed: bool,
        solar_energy_input: float,
        weather_conditions: WeatherConditions,
        internal_resolution: float,
        air_gap_thickness: float,
        pv_to_collector_thermal_conductance: float,
        glass_temperature: Optional[float],
        glass_emissivity: Optional[float],
        collector_temperature: float,
    ) -> Tuple[float, float]:
        """
        Update the internal properties of the PV layer based on external factors.

        :param glazed:
            Whether or not the panel is glazed, I.E., whether the panel has a glass
            layer or not.

        :param solar_energy_input:
            The solar irradiance, normal to the panel, measured in Joules per meter
            sqaured per time interval.

        :param weather_conditions:
            The current weather conditions, passed in as a :class:`WeatherConditions`
            instance.

        :param internal_resolution:
            The resolution of the model being run, measured in seconds.

        :param air_gap_thickness:
            The thickness of the gap between the glass and PV layers, measured in
            meters.

        :param pv_to_collector_thermal_conductance:
            The thermal conductance between the PV and collector layers, measured in
            Watts per meter squared Kelvin.

        :param glass_temperature:
            The temperature glass layer of the PV-T panel in Kelvin.

        :param glass_emissivity:
            The emissivity of the glass layer.

        :param collector_temperature:
            The temperature of the collector layer, measured in Kelvin.

        :return:
            The heat transferred to the collector and the glass layers respectively as a
            `Tuple`. Both these values are measured in Joules.

        """

        # pdb.set_trace(header="PV Module's Update Method.")

        # Determine the excess heat that has been inputted into the panel during this
        # time step, measured in Joules.
        solar_heat_input = _solar_heat_input(
            solar_energy_input,
            self.area,
            self.efficiency,
            self._absorptivity,
            self._transmissivity,
        )  # [J] or [J/time_step]

        if glazed:
            radiative_loss_upwards = (
                layer_to_glass_radiative_transfer(
                    glass_temperature,  # type: ignore
                    glass_emissivity,  # type: ignore
                    self.area,
                    self.temperature,
                    self.emissivity,
                )  # [W]
                * internal_resolution  # [seconds]
            )  # [J]

            convective_loss_upwards = (
                layer_to_glass_conductive_transfer(
                    air_gap_thickness,
                    glass_temperature,  # type: ignore
                    self.area,
                    self.temperature,
                )  # [W]
                * internal_resolution  # [seconds]
            )  # [J]
        else:
            # @@@ The PV layer currently works to only emit as a full layer to the sky.
            radiative_loss_upwards = (
                self._layer_to_sky_radiative_transfer(
                    fraction_emitting=1,
                    sky_temperature=weather_conditions.sky_temperature,
                )  # [W]
                * internal_resolution  # [seconds]
            )  # [J]

            convective_loss_upwards = (
                self._layer_to_air_convective_transfer(
                    weather_conditions.ambient_temperature,
                    fraction_emitting=1,
                    wind_heat_transfer_coefficient=weather_conditions.wind_heat_transfer_coefficient,
                )  # [W]
                * internal_resolution  # [seconds]
            )  # [J]

        # This value is returned in Watts, and so needs to be multiplied by the time
        # interval.
        pv_to_collector = (
            _pv_to_collector_conductive_transfer(
                self.area,
                pv_to_collector_thermal_conductance,
                self.temperature,
                collector_temperature,
            )  # [W]
            * internal_resolution  # [seconds]
        )  # [J]

        heat_lost = (
            radiative_loss_upwards + convective_loss_upwards + pv_to_collector
        )  # [J]

        # Use this to compute the rise in temperature of the PV layer and set the
        # temperature appropriately.
        self.temperature += (solar_heat_input - heat_lost) / (  # [J]
            self._mass * self._heat_capacity  # [kg]  # [J/kg*K]
        )  # [K]

        # Return the heat transfered to the glass and collector layers.
        return (
            pv_to_collector,  # [J]
            (
                (radiative_loss_upwards + convective_loss_upwards)  # [J] + [J]
                / (internal_resolution)  # [seconds]
            ),  # [W]
        )

    @property
    def efficiency(self) -> float:
        """
        Returns the percentage efficiency of the PV panel based on its temperature.

        :return:
            A decimal giving the percentage efficiency of the PV panel between 0 (0%
            efficiency), and 1 (100% efficiency).

        """

        return self._reference_efficiency * (  # [unitless]
            1
            - self._thermal_coefficient  # [1/K]
            * (self.temperature - self._reference_temperature)  # [K]
        )


class Collector(_OpticalLayer):
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
        self._number_of_pipes = collector_params.number_of_pipes
        self.output_water_temperature = collector_params.output_water_temperature
        self._mass_flow_rate = collector_params.mass_flow_rate
        self._pipe_diameter = collector_params.pipe_diameter
        self.htf_heat_capacity = collector_params.htf_heat_capacity
        self.pump_power = collector_params.pump_power

    @property
    def mass_flow_rate(self) -> float:
        """
        Return the mass-flow rate in kilograms per second.

        :return:
            d/dt(M) in kg/s

        """

        return self._mass_flow_rate / (3600)  # [kg/s]

    def __repr__(self) -> str:
        """REPR"""

        return (
            "Collector(mass: {}kg, heat_capacity: {}J/kg*K, area: {}/m^2, ".format(
                self._mass,
                self._heat_capacity,
                self.area,
            )
            + "thickness: {}/m, ".format(
                self.thickness,
            )
            + "temp: {}/K, output_temp: {}/K, mass_flow_rate: {}/kg/s, ".format(
                self.temperature, self.output_water_temperature, self.mass_flow_rate
            )
            + "htf_heat_capacity: {}/J/kg*K)".format(self.htf_heat_capacity)
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

    def update(
        self,
        internal_resolution: float,
        air_gap_thickness: float,
        glass_layer_included: bool,
        glass_temperature: Optional[float],
        glass_emissivity: Optional[float],
        pv_layer_included: bool,
        portion_covered: float,
        collector_heat_input: float,
        input_water_temperature: float,
        weather_conditions: WeatherConditions,
        back_plate: BackPlate,
    ) -> Tuple[float, float]:
        """
        Update the internal properties of the PV layer based on external factors.

        :param internal_resolution:
            The resolution at which the simulation is being run, measured in seconds.

        :param air_gap_thickness:
            The thickness, measured in meters, of the air gap between the glass layer
            and the rest of the panel. This parameter is only needed when there is no PV
            layer but the thermal collector is glazed. In this case, the collector layer
            experiences a radiative heat loss to the glass layer.

        :param glass_layer_included:
            Whether there is a glass layer present, radiating to the sky, or whether the
            collector layer (or PV layer) is exposed directly to the sky.

        :param glass_temperature:
            The temperature of the glass layer, if present, measured in Kelvin. If there
            is no glass layer present, then this is None.

        :param glass_emissivity:
            The emissivity of the glass layer, if present. If no glass layer is present,
            then this is None.

        :param pv_layer_included:
            Whether there is a PV layer in the panel acting as the solar absorber, or
            whether the collector layer directly absorbs incident sunglight.

        :param portion_covered:
            The portion of the PV-T panel which is covered with PV.

        :param pv_temperature:
            The temperature of the PV layer of the panel, measured in Kelvin.

        :param collector_heat_input:
            Heat inputted to the collector layer, measured in Joules.

        :param input_water_temperature:
            The temperature of the input water flow to the collector, measured in
            Kelvin, at the current time step.

        :param weather_conditions:
            The current weather conditions.

        :param back_plate:
            The back plate of the PV-T panel, through which heat is lost.

        :return:
            The output water temperature from the collector.

        """

        if self.temperature > 1000:
            pdb.set_trace(
                header="Thermal collector melting - temperature greater than 1000K."
            )

        # From the excess heat, compute what is not lost to the environment, and, from
        # there, what is transferred to the HTF.
        back_plate_heat_loss = (
            back_plate.conductance  # [W/m^2*K]
            * self.area  # [m^2]
            * (self.temperature - weather_conditions.ambient_temperature)  # [K]
        ) * internal_resolution  # [seconds]  # [J]

        # If there are no glass or PV layers, then we lose heat from the collector
        # layer directly.
        if not glass_layer_included and (not pv_layer_included or portion_covered != 1):
            upward_heat_losses: float = (
                self._layer_to_air_convective_transfer(
                    weather_conditions.ambient_temperature,
                    (1 - portion_covered),
                    weather_conditions.wind_heat_transfer_coefficient,
                )
                * internal_resolution
                + self._layer_to_sky_radiative_transfer(
                    (1 - portion_covered), weather_conditions.sky_temperature
                )
                * internal_resolution
            )  # [J]
        # If there is a glass layer, and a PV layer that does not fully cover the panel,
        # then we need to compute the energy transferred to the glass layer.
        elif glass_layer_included and (not pv_layer_included or portion_covered != 1):
            if glass_temperature is None or glass_emissivity is None:
                raise ProgrammerJudgementFault(
                    "The system attempted to compute a radiative and/or conductive "
                    "transfer to a non-existant glass layer."
                )
            upward_heat_losses = (
                layer_to_glass_radiative_transfer(
                    glass_temperature,
                    glass_emissivity,
                    self.area * (1 - portion_covered),
                    self.temperature,
                    self.emissivity,
                )
                * internal_resolution
                + layer_to_glass_conductive_transfer(
                    air_gap_thickness,
                    glass_temperature,
                    self.area * (1 - portion_covered),
                    self.temperature,
                )
                * internal_resolution
            )  # [J]
        else:
            upward_heat_losses = 0  # [J]

        # Compute the output water temperature
        self.output_water_temperature = (
            self.convective_heat_transfer_coefficient_of_water  # [W/m^2*K]
            * self.temperature  # [K]
            * self.htf_surface_area  # [m^2]
            + (
                self.mass_flow_rate * self.htf_heat_capacity  # [kg/s]  # [J/kg*K]
                - 0.5
                * self.convective_heat_transfer_coefficient_of_water  # [W/m^2/K]
                * self.htf_surface_area  # [m^2]
            )
            * input_water_temperature  # [K]
        ) / (  # [W]
            self.mass_flow_rate * self.htf_heat_capacity  # [kg/s]  # [J/kg*K]
            + 0.5
            * self.convective_heat_transfer_coefficient_of_water  # [W/m^2/K]
            * self.htf_surface_area  # [m^2]
        )  # [W/K]

        # Compute the heat loss to the bulk water
        bulk_water_heat_loss = (
            self.mass_flow_rate  # [kg/s]
            * internal_resolution  # [seconds]
            * self.htf_heat_capacity  # [J/kg*K]
            * (self.output_water_temperature - input_water_temperature)  # [K] - [K]
        )  # [J]

        # This heat is now converted into Joules.
        net_heat_gain = collector_heat_input - (  # [J]
            back_plate_heat_loss  # [J]
            + upward_heat_losses  # [J]
            + bulk_water_heat_loss  # [J]
        )
        self.temperature += net_heat_gain / (self._mass * self._heat_capacity)

        return (
            self.output_water_temperature,  # [K]
            upward_heat_losses,  # [W]
        )


class PVT:
    """
    Represents an entire PV-T collector.

    .. attribute:: timezone
        The timezone that the PVT system is based in.

    """

    # Private attributes:
    #
    # .. attribute:: _glass
    #   Represents the upper (glass) layer of the panel. This is set to `None` if the
    #   panel is unglazed.
    #
    # .. attribute:: _air_gap_thickness
    #   The thickness of the air gap between the glass and PV (or collector) layers,
    #   measured in meters.
    #
    # .. attribute:: _portion_covered
    #   The portion of the PV-T panel which is covered with PV.
    #
    # .. attribute:: _pv_to_collector_thermal_conductance
    #   The thermal conductance, in Watts per meter squared Kelvin, between the PV layer
    #   and collector layer of the panel.
    #
    # .. attribute:: _pv
    #   Represents the middle (pv) layer of the panel. Can be set to `None` if not
    #   present in the panel.
    #
    # .. attribute:: _collector
    #   Represents the lower (thermal-collector) layer of the panel.
    #
    # .. attribute:: _vertical_tracking
    #   A `bool` giving whether or not the panel tracks verticallly.
    #
    # .. attribute:: _tilt
    #   The angle between the normal to the panel's surface and the horizontal.
    #
    # .. attribute:: _horizontal_tracking
    #   A `bool` giving whether or not the panel tracks horizontally.
    #
    # .. attribute:: _azimuthal_orientation
    #   The angle between the normal to the panel's surface and True North.
    #

    def __init__(
        self,
        latitude: float,
        longitude: float,
        area: float,
        glazed: bool,
        glass_parameters: OpticalLayerParameters,
        collector_parameters: CollectorParameters,
        back_params: BackLayerParameters,
        air_gap_thickness: float,
        portion_covered: float,
        pv_to_collector_thermal_conductance: float,
        timezone: datetime.timezone,
        *,
        pv_layer_included: bool = False,
        pv_parameters: Optional[PVParameters],
        tilt: Optional[float] = None,
        azimuthal_orientation: Optional[float] = None,
        horizontal_tracking: bool = False,
        vertical_tracking: bool = False,
    ) -> None:
        """
        Instantiate an instance of the PV-T collector class.

        :param latitude:
            The latitude of the PV-T system, defined in degrees relative to the equator
            in the standard way.

        :param longitude:
            The longitude of the PV-T system, defined in degrees relative to the
            Greenwich meridian.

        :param area:
            The area of the panel, measured in meters squared.

        :param glazed:
            Whether the panel is glazed. I.E., whether the panel has a glass layer.

        :param glass_parameters:
            Parameters used to instantiate the glass layer.

        :param collector_parameters:
            Parametsrs used to instantiate the collector layer.

        :param back_params:
            Parameters used to instantiate the back layer.

        :param air_gap_thickness:
            The thickness, in meters, of the air gap between the PV and glass layers.

        :param portion_covered:
            The portion of the PV-T panel which is covered with PV, ranging from 0
            (none) to 1 (all).

        :param pv_layer_included:
            Whether or not the system includes a photovoltaic layer.

        :param pv_parameters:
            Parameters used to instantiate the PV layer.

        :param tilt:
            The tilt of the panel above the horizontal, measured in degrees.

        :param azimuthal_orientation:
            The orientation of the surface of the panel, defined relative to True North,
            measured in degrees.

        :param horizontal_tracking:
            Whether or not the PV-T system tracks the sun horizontally.

        :param vertical_tracking:
            Whether or not the PV-T system tracks the sun vertically.

        """

        # If the PV layer parameters have not been specified, then raise an error.
        if pv_layer_included and pv_parameters is None:
            raise MissingParametersError(
                "PVT",
                "PV-layer parameters must be provided if including a PV layer.",
            )

        self.area = area

        # If the orientation parameters have not been specified correctly, then raise an
        # error.
        if not vertical_tracking and tilt is None:
            raise MissingParametersError(
                "PVT",
                "If the panel does not track vertically, then the tilt must be given.",
            )
        if not horizontal_tracking and azimuthal_orientation is None:
            raise MissingParametersError(
                "PVT",
                "If the panel does not track horizontally, then the azimuthal "
                "orientation must be given.",
            )

        # * Log if too many parameters were provided/specified :) .

        if glazed:
            self._glass: Optional[Glass] = Glass(glass_parameters)
        else:
            self._glass = None

        self._air_gap_thickness = air_gap_thickness
        self._portion_covered = portion_covered
        self._pv_to_collector_thermal_conductance = pv_to_collector_thermal_conductance

        if pv_layer_included and pv_parameters is not None:
            pv_parameters.area *= portion_covered
            self._pv: Optional[PV] = PV(pv_parameters)
        else:
            self._pv = None

        self._collector = Collector(collector_parameters)

        self._back_plate = BackPlate(back_params)

        self._latitude = latitude
        self._longitude = longitude

        self._vertical_tracking = vertical_tracking
        self._tilt = tilt
        self._horizontal_tracking = horizontal_tracking
        self._azimuthal_orientation = azimuthal_orientation

        self.timezone = timezone

    def _get_solar_orientation_diff(
        self, declination: float, azimuthal_angle: float
    ) -> float:
        """
        Determine the between the panel's normal and the sun.

        :param declination:
            The declination of the sun: the angle in degrees of the sun above the
            horizon.

        :param azimuthal_angle:
            The position of the sun, defined as degrees clockwise from True North.

        :return:
            The angle in degrees between the solar irradiance and the normal to the
            panel.

        """

        # Determine the angle between the sun and panel's normal, both horizontally and
        # vertically. If tracking is enabled, then this angle should be zero along each
        # of those axes. If tracking is disabled, then this angle is just the difference
        # between the panel's orientation and the sun's.

        if self._horizontal_tracking or self._azimuthal_orientation is None:
            horizontal_diff: float = 0
        else:
            horizontal_diff = abs(self._azimuthal_orientation - azimuthal_angle)

        if self._vertical_tracking or self._tilt is None:
            vertical_diff: float = 0
        else:
            vertical_diff = abs(self._tilt - declination)

        # Combine these to generate the angle between the two directions.
        return math.degrees(
            math.acos(
                math.cos(math.radians(horizontal_diff))
                * math.cos(math.radians(vertical_diff))
            )
        )

    def get_solar_irradiance(self, weather_conditions: WeatherConditions) -> float:
        """
        Returns the irradiance striking the panel normally in Watts per meter squared.

        :param weather_conditions:
            The current weather conditions.

        :return:
            The solar irradiance striking the panel, adjusted for the sun's declination
            and azimuthal angle, as well as the panel's orientation, dependant on the
            panel's latitude and longitude.

        """

        # Compute the angle of solar irradiance wrt the panel.
        solar_diff_angle = self._get_solar_orientation_diff(
            weather_conditions.declination, weather_conditions.azimuthal_angle
        )

        return (
            weather_conditions.irradiance  # [W/m^2]
            * math.cos(math.radians(solar_diff_angle))
            if weather_conditions.declination >= MINIMUM_SOLAR_DECLINATION
            and 0 <= solar_diff_angle <= MAXIMUM_SOLAR_DIFF_ANGLE
            else 0
        )

    def update(
        self,
        input_water_temperature: float,
        internal_resolution: float,
        weather_conditions: WeatherConditions,
    ) -> float:
        """
        Updates the properties of the PV-T collector based on a changed input temp..

        :param input_water_temperature:
            The water temperature going into the PV-T collector.

        :param internal_resolution:
            The resolution of the model being run, measured in seconds.

        :param weather_conditions:
            The weather conditions at the time of day being incremented to.

        :return:
            The output water temperature from the PV-T panel.

        """

        # pdb.set_trace(header="Debug PVT Module.")

        # Compute the solar energy inputted to the system in Joules per meter squared.
        solar_energy_input = (
            self.get_solar_irradiance(weather_conditions)  # [W/m^2]
            * internal_resolution  # [seconds]
        )  # [J/m^2]
        # @@@ The units of the solar_energy_input variable will be dependant on the time
        # @@@ step, and so aren't purely Joules, but, rather, Watts.time_step, or smthg.

        # Call the pv panel to update its temperature.
        pv_to_glass_heat_input: Optional[float] = None
        if self._pv is not None:
            # The excese PV heat generated is in units of Watts.
            collector_heat_input, pv_to_glass_heat_input = self._pv.update(
                self.glazed,
                solar_energy_input,
                weather_conditions,
                internal_resolution,
                self._air_gap_thickness,
                self._pv_to_collector_thermal_conductance,
                self._glass.temperature if self._glass is not None else None,
                self._glass.emissivity if self._glass is not None else None,
                self._collector.temperature,
            )  # [J], [W]
            collector_heat_input += (
                solar_energy_input  # [W/m^2]
                * (1 - self._portion_covered)
                * self.area  # [m^2]
                * internal_resolution  # [s]
            )  # [J]
        else:
            collector_heat_input = (
                solar_energy_input * self.area  # [J/time_step*m^2] * [m^2]
            )  # [J]
            pv_to_glass_heat_input = None  # [W]

        # Based on the heat supplied, both from the sun (depending on whether there is
        # no PV layer present, or whether the PV layer does not fully cover the panel),
        # and from the heat transfered in from the PV layer.
        (
            output_water_temperature,  # [K]
            collector_to_glass_heat_input,  # [W]
        ) = self._collector.update(
            internal_resolution,
            self._air_gap_thickness,
            self._glass is not None,
            self._glass.temperature if self._glass is not None else None,
            self._glass.emissivity if self._glass is not None else None,
            self._pv is not None,
            self._portion_covered,
            collector_heat_input,
            input_water_temperature,
            weather_conditions,
            self._back_plate,
        )

        # Determine the heat inputted to the glass layer.
        glass_heat_input: float = 0
        if pv_to_glass_heat_input is not None:
            glass_heat_input += pv_to_glass_heat_input
        if collector_to_glass_heat_input is not None:
            glass_heat_input += collector_to_glass_heat_input
        else:
            raise ProgrammerJudgementFault(
                "The panel has neither a PV or Collector layer... Pretty useless panel."
            )

        # Pass this new temperature through to the glass instance to update it.
        if self._glass is not None:
            self._glass.update(
                internal_resolution,
                self._air_gap_thickness,
                self.area,
                weather_conditions,
                glass_heat_input,
            )

        return output_water_temperature

    @property
    def glazed(self) -> bool:
        """
        Returns whether the panel is glazed, ie whether it has a glass layer.

        :return:
            Whether the panel is glazed (True) or unglazed (False).

        """

        return self._glass is not None

    @property
    def electrical_efficiency(self) -> Optional[float]:
        """
        Returns the electrical efficiency of the PV-T system.

        :return:
            The electrical efficiency of the PV layer. `None` is returned if no PV layer
            is present.

        """

        if self._pv is None:
            return None

        return self._pv.efficiency

    @property
    def glass_temperature(self) -> Optional[float]:
        """
        Returns the temperature of the glass layer of the PV-T system.

        :return:
            The temperature, in Kelvin, of the glass layer of the PV-T system.

        """

        return self._glass.temperature if self._glass is not None else None

    @property
    def pv_temperature(self) -> Optional[float]:
        """
        Returns the temperature of the PV layer of the PV-T system.

        :return:
            The temperature in Kelvin of the PV layer of the PV-T system. If no PV layer
            is installed, then `None` is returned.

        """

        if self._pv is None:
            return None

        return self._pv.temperature

    @property
    def collector_temperature(self) -> float:
        """
        Returns the temperature of the collector layer of the PV-T system.

        :return:
            The temperature, in Kelvin, of the collector layer of the PV-T system.

        """

        return self._collector.temperature

    @property
    def output_water_temperature(self) -> float:
        """
        Returns the temperature of the water outputted by the panel, measured in Kelvin.

        :return:
            The output water temperature from the thermal collector.

        """

        return self._collector.output_water_temperature

    @property
    def coordinates(self) -> Tuple[float, float]:
        """
        Returns a the coordinates of the panel.

        :return:
            A `tuple` containing the latitude and longitude of the panel.

        """

        return (self._latitude, self._longitude)

    @property
    def mass_flow_rate(self) -> float:
        """
        Returns the mass flow rate through the collector, measured in kg/s.

        :return:
            The mass flow rate of heat-transfer fluid through the thermal collector.

        """

        return self._collector.mass_flow_rate

    @property
    def htf_heat_capacity(self) -> float:
        """
        Returns the heat-capacity of the heat-transfer fluid in the collector.

        :return:
            The heat-capacity of the HTF of the collector, measured in Joules per
            kilogram Kelvin.

        """

        return self._collector.htf_heat_capacity

    @property
    def collector_output_temperature(self) -> float:
        """
        Returns the output temperature, in Kelvin, of HTF from the collector.

        :return:
            The collector output temp in Kelvin.

        """

        return self._collector.output_water_temperature

    @property
    def pump_power(self) -> float:
        """
        Returns the power, in Watts, consumed by the HTF pump.

        :return:
            The water-pump power consumption in Watts.

        """

        return self._collector.pump_power

    def electrical_output(self, weather_conditions: WeatherConditions) -> float:
        """
        Returns the electrical output of the PV-T panel in Watts.

        NOTE: We here need to include the portion of the panel that is covered s.t. the
        correct electricitiy-generating area is accounted for, rather than accidentailly
        inculding areas which do not generated electricity.

        :param weather_conditions:
            The current weather conditions at the time step being incremented to.

        :return:
            The electrical output, in Watts, of the PV-T panel.

        """

        return (
            self.electrical_efficiency
            * self.get_solar_irradiance(weather_conditions)
            * self.area
            * self._portion_covered
            if self.electrical_efficiency is not None
            else 0
        )

    def __repr__(self) -> str:
        """
        A nice-looking representation of the PV-T panel.

        :return:
            A nice-looking representation of the PV-T panel.

        """

        return "PVT(coordinates: {}N {}E, tilt: {}, azimuthal_orientation: {}".format(
            self._latitude, self._longitude, self._tilt, self._azimuthal_orientation
        ) + "_glass: {}, _pv: {}, _collector: {}, _back_plate: {}_".format(
            self._glass, self._pv, self._collector, self._back_plate
        )