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

from typing import Optional, Tuple

from .__utils__ import (
    MissingParametersError,
    LayerParameters,
    BackLayerParameters,
    CollectorParameters,
    OpticalLayerParameters,
    PVParameters,
    WeatherConditions,
    STEFAN_BOLTZMAN_CONSTANT,
    THERMAL_CONDUCTIVITY_OF_AIR,
    FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR,
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
    solar_irradiance: float,
    pv_area: float,
    pv_efficiency: float,
    pv_absorptivity: float,
    pv_transmissivity: float,
) -> float:
    """
    Determines the heat input due to solar irradiance.

    :param solar_irradiance:
        The solar irradiance, normal to the panel, measured in Energy per meter
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
        * solar_irradiance
        * pv_area
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
        The mass-flow rate of heat-transfer fluid through the collector.

    :param collector_htf_heat_capacity:
        The heat capacity of the heat-transfer fluid flowing through the collector,
        measured in Joules per kilogram Kelvin.

    :return:
        The heat transfer, in Watts, to the heat transfer fluid.

    """

    return (
        collector_mass_flow_rate
        * collector_htf_heat_capacity
        * (collector_temperature - input_water_temperature)
    )


def _pv_to_collector_conductive_transfer(
    ambient_temperature: float,
    resolution: float,
    input_water_temperature: float,
    collector_temperature: float,
    collector_mass_flow_rate: float,
    collector_htf_heat_capacity: float,
    back_plate_conductance: float,
) -> float:
    """
    Computes the heat transfer from the PV layer to the thermal collector.

    :param ambient_temperature:
        The ambient temperature in Kelvin.

    :param resolution:
        The resolution of the simulation being run, measured in minutes.

    :param input_water_temperature:
        The input water temperature to the PV-T panel, measured in Kelvin.

    :param collector_temperature:
        The temperature of the collector layer, measured in Kelvin.

    :param collector_mass_flow_rate:
        The mass-flow rate of heat-transfer fluid through the collector.

    :param collector_htf_heat_capacity:
        The heat capacity of the heat-transfer fluid flowing through the collector,
        measured in Joules per kilogram Kelvin.

    :param back_plate_conductance:
        The conductance of the back plate of the panel.

    :return:
        The heat transfer, in Watts, from the PV layer to the thermal-collector
        layer.

    """

    back_plate_heat_loss = back_plate_conductance * (
        collector_temperature - ambient_temperature
    )

    # It's necessary to divide by the model resolution to convert from the resolution
    # time-step to Watts.
    htf_heat_transfer = (
        _htf_heat_transfer(
            input_water_temperature,
            collector_temperature,
            collector_mass_flow_rate,
            collector_htf_heat_capacity,
        )
        / (resolution * 60)
    )

    return back_plate_heat_loss + htf_heat_transfer


def pv_to_glass_radiative_transfer(
    glass_temp: float,
    glass_emissivity: float,
    pv_area: float,
    pv_temperature: float,
    pv_emissivity: float,
) -> float:
    """
    Computes the radiative heat transfer from the PV layer to the glass layer.

    :param glass_temp:
        The temperature of the glass layer of the PV-T system.

    :param glass_emissivity:
        The emissivity of the glass layer of the PV-T system.

    :param pv_area:
        The area of the PV layer (and the entier PV-T system), measured in meters
        squared.

    :param pv_temperature:
        The temperature of the PV layer, measured in Kelvin.

    :param pv_emissivity:
        The emissivity of the PV layer, defined between 0 and 1.

    :return:
        The heat transfer, in Watts, from the PV layer to the glass layer that takes
        place by radiative transfer.

    """

    return (
        STEFAN_BOLTZMAN_CONSTANT * pv_area * (pv_temperature ** 4 - glass_temp ** 4)
    ) / (1 / pv_emissivity + 1 / glass_emissivity - 1)


def pv_to_glass_conductive_transfer(
    air_gap_thickness: float, glass_temp: float, pv_area: float, pv_temperature: float
) -> float:
    """
    Computes the conductive heat transfer between the PV layer and glass cover.

    :param air_gap_thickness:
        The thickness of the air gap between the PV and glass layers.

    :param glass_temp:
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
        THERMAL_CONDUCTIVITY_OF_AIR
        * (pv_temperature - glass_temp)
        * pv_area
        / air_gap_thickness
    )


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

        self._mass = layer_params.mass
        self._heat_capacity = layer_params.heat_capacity
        self.temperature = (
            layer_params.temperature if layer_params.temperature is not None else 273
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
                self._mass, self._heat_capacity, self.area, self.thickness
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
            self.thickness / self.conductivity
            + 1 / FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR
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
                optical_params.mass,
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
                self._mass, self._heat_capacity, self.area, self.thickness
            )
            + "temp: {}, transmissivity: {}, absorptivity: {}, ".format(
                self.temperature, self._transmissivity, self._absorptivity
            )
            + "emissivitiy: {})".format(self.emissivity)
        )


class Glass(_OpticalLayer):
    """
    Represents the glass (upper) layer of the PV-T panel.

    """

    def _layer_to_sky_radiative_transfer(self, sky_temperature: float) -> float:
        """
        Calculates the heat loss to the sky radiatively from the layer.

        :param sky_temperature:
            The radiative temperature of the sky, measured in Kelvin.

        :return:
            The heat transfer, in Watts per meter squared, radiatively to the sky.

        """

        return (
            self.emissivity
            * STEFAN_BOLTZMAN_CONSTANT
            * (self.temperature ** 4 - sky_temperature)
        )

    # @@@ The unused-import flag is disabled as wind_speed will be included eventually
    def _layer_to_air_conductive_transfer(
        self,
        ambient_temperature: float,
        wind_speed: float,  # pylint: disable=unused-import
    ) -> float:
        """
        Calculates the heat loss to the surrounding air by conduction and convection.

        :param ambient_temperature:
            The ambient temperature of the air, measured in Kelvin.

        :param wind_speed:
            The wind speed in meters per second.

        :return:
            The heat transfer, in Watts per meter squared, conductively to the
            surrounding air.

        """

        # @@@ Include Suresh Kumar Wind heat transfer coefficient in solar collectors in
        # @   outdoor conditions here to better estimate h_wind.
        wind_heat_transfer_coefficient = 10

        return wind_heat_transfer_coefficient * (self.temperature - ambient_temperature)

    def update(
        self,
        air_gap_thickness: float,
        pv_area: float,
        pv_temperature: float,
        pv_emissivity: float,
        weather_conditions: WeatherConditions,
    ) -> None:
        """
        Update the internal properties of the PV layer based on external factors.

        :param air_gap_thickness:
            The thickness, in meters, of the air gap between the PV and glass layers.

        :param pv_area:
            The area of the PV panel, measured in meters squared.

        :param pv_temperature:
            The temperature of the PV panel, measured in Kelvin.

        :param pv_emissivity:
            The emissivity of the PV panel, defined between 0 (no light is emitted) and
            1 (all incident light is re-emitted).

        :param weather_conditions:
            The weather conditions at the current time step.

        """

        # Set the temperature of this layer appropriately.
        heat_losses = self._layer_to_air_conductive_transfer(
            weather_conditions.ambient_temperature, weather_conditions.wind_speed
        ) + self._layer_to_sky_radiative_transfer(weather_conditions.sky_temperature)

        heat_input = pv_to_glass_conductive_transfer(
            air_gap_thickness, self.temperature, pv_area, pv_temperature
        ) + pv_to_glass_radiative_transfer(
            self.temperature, self.emissivity, pv_area, pv_temperature, pv_emissivity
        )

        excess_heat = heat_input - heat_losses

        self.temperature = self.temperature + excess_heat / (
            self._mass * self._heat_capacity
        )


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
                self._mass, self._heat_capacity, self.area, self.thickness
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

    def excess_heat(
        self,
        solar_irradiance: float,
        input_water_temperature: float,
        ambient_temperature: float,
        resolution: float,
        air_gap_thickness: float,
        glass_temp: float,
        glass_emissivity: float,
        collector_temperature: float,
        collector_mass_flow_rate: float,
        collector_htf_heat_capacity: float,
        back_plate_conductance: float,
    ) -> float:
        """
        Computes the excess heat provided to the PV layer.

        :param solar_irradiance:
            The solar irradiance, normal to the panel, measured in Watts per meter
            squared.

        :param input_water_temperature:
            The input water temperature to the panel, measured in Kelvin.

        :param ambient_temperature:
            The ambient temperature of the air surrounding the panel, measured in
            Kelvin.

        :param resolution:
            The resolution of the simulation being run, measured in minutes.

        :param air_gap_thickness:
            The thickness of the gap between the glass and PV layers, measured in
            meters.

        :param glass_temp:
            The temperature glass layer of the PV-T panel in Kelvin.

        :param glass_emissivity:
            The emissivity of the glass layer.

        :param collector_temperature:
            The temperature of the collector layer of the PV-T panel.

        :param collector_mass_flow_rate:
            The mass-flow rate of heat-transfer fluid through the collector layer of the
            PV-T panel.

        :param collector_htf_heat_capacity:
            The heat capacity of the heat-transfer fluid flowing through the collector,
            measured in Joules per kilogram Kelvin.

        :param back_plate_conductance:
            The conductante of the back plate of the PV-T panel.

        :return:
            The excess heat after balancing the heat inputs and outputs for the layer.

        """

        solar_heat_input = _solar_heat_input(
            solar_irradiance,
            self.area,
            self.efficiency,
            self._absorptivity,
            self._transmissivity,
        )

        pv_to_glass_rad = (
            self.area
            * pv_to_glass_radiative_transfer(
                glass_temp,
                glass_emissivity,
                self.area,
                self.temperature,
                self.emissivity,
            )
            * resolution
            * 60
        )

        pv_to_glass_cond = (
            pv_to_glass_conductive_transfer(
                air_gap_thickness, glass_temp, self.area, self.temperature
            )
            * resolution
            * 60
        )

        pv_to_collector = (
            _pv_to_collector_conductive_transfer(
                ambient_temperature,
                resolution,
                input_water_temperature,
                collector_temperature,
                collector_mass_flow_rate,
                collector_htf_heat_capacity,
                back_plate_conductance,
            )
            * resolution
            * 60
        )

        return solar_heat_input - pv_to_glass_rad - pv_to_glass_cond - pv_to_collector

    def update(
        self,
        solar_irradiance: float,
        input_water_temperature: float,
        ambient_temperature: float,
        resolution: float,
        air_gap_thickness: float,
        glass_temp: float,
        glass_emissivity: float,
        collector_temperature: float,
        collector_mass_flow_rate: float,
        collector_htf_heat_capacity: float,
        back_plate_conductance: float,
    ) -> float:
        """
        Update the internal properties of the PV layer based on external factors.

        :param solar_irradiance:
            The solar irradiance, normal to the panel, measured in Joules per meter
            sqaured per time interval.

        :param input_water_temperature:
            The input water temperature to the panel, measured in Kelvin.

        :param ambient_temperature:
            The ambient temperature of the air surrounding the panel, measured in
            Kelvin.

        :param resolution:
            The resolution of the model being run, measured in minutes.

        :param air_gap_thickness:
            The thickness of the gap between the glass and PV layers, measured in
            meters.

        :param glass_temp:
            The temperature glass layer of the PV-T panel in Kelvin.

        :param glass_emissivity:
            The emissivity of the glass layer.

        :param collector_temperature:
            The temperature of the collector layer, measured in Kelvin.

        :param collector_mass_flow_rate:
            The mass-flow rate of heat-transfer fluid through the collector.

        :param collector_htf_heat_capacity:
            The heat capacity of the heat-transfer fluid flowing through the collector,
            measured in Joules per kilogram Kelvin.

        :param back_plate_conductance:
            The conductance of the back-plate of the PV-T panel.

        :return:
            The excess heat provided which is transferred to the thermal collector. This
            value is returned in Watts.

        """

        # Determine the excess heat that has been inputted into the panel during this
        # time step, measured in Joules.
        excess_heat = self.excess_heat(
            solar_irradiance,
            input_water_temperature,
            ambient_temperature,
            resolution,
            air_gap_thickness,
            glass_temp,
            glass_emissivity,
            collector_temperature,
            collector_mass_flow_rate,
            collector_htf_heat_capacity,
            back_plate_conductance,
        )

        # Use this to compute the rise in temperature of the PV layer and set the
        # temperature appropriately.
        self.temperature = self.temperature + excess_heat / (
            self._mass * self._heat_capacity * resolution * 60
        )

        # Return the excess heat, transferred to the collector, based on the new PV-T
        # temperature.
        return self.excess_heat(
            solar_irradiance,
            input_water_temperature,
            ambient_temperature,
            resolution,
            air_gap_thickness,
            glass_temp,
            glass_emissivity,
            collector_temperature,
            collector_mass_flow_rate,
            collector_htf_heat_capacity,
            back_plate_conductance,
        )

    @property
    def efficiency(self) -> float:
        """
        Returns the percentage efficiency of the PV panel based on its temperature.

        :return:
            A decimal giving the percentage efficiency of the PV panel between 0 (0%
            efficiency), and 1 (100% efficiency).

        """

        return self._reference_efficiency * (
            1
            - self._thermal_coefficient
            * (self.temperature - self._reference_temperature)
        )


class Collector(_Layer):
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
            LayerParameters(
                collector_params.mass,
                collector_params.heat_capacity,
                collector_params.area,
                collector_params.thickness,
                collector_params.temperature,
            )
        )

        self.output_water_temperature = collector_params.output_water_temperature
        self._mass_flow_rate = collector_params.mass_flow_rate
        self.htf_heat_capacity = collector_params.htf_heat_capacity

    @property
    def mass_flow_rate(self) -> float:
        """
        Return the mass-flow rate in kilograms per second.

        :return:
            d/dt(M) in kg/s

        """

        return self._mass_flow_rate / (3600)

    def __repr__(self) -> str:
        """REPR"""

        return (
            "Collector(mass: {}, heat_capacity: {}, area: {}, thickness: {}, ".format(
                self._mass, self._heat_capacity, self.area, self.thickness
            )
            + "temp: {}, output_temp: {}, mass_flow_rate: {}, ".format(
                self.temperature, self.output_water_temperature, self.mass_flow_rate
            )
            + "htf_heat_capacity: {})".format(self.htf_heat_capacity)
        )

    def update(
        self,
        resolution: int,
        excess_pv_heat: float,
        input_water_temperature: float,
        ambient_temperature: float,
        back_plate: BackPlate,
    ) -> float:
        """
        Update the internal properties of the PV layer based on external factors.

        :param resolution:
            The resolution at which the simulation is being run, measured in minutes.

        :param excess_pv_heat:
            Excess heat from the adjoining PV layer, measured in Watts.

        :param input_water_temperature:
            The temperature of the input water flow to the collector, measured in
            Kelvin, at the current time step.

        :param ambient_temperature:
            The ambient temperature surrounding the panel.

        :param back_plate:
            The back plate of the PV-T panel, through which heat is lost.

        :return:
            The output water temperature from the collector.

        """

        if self.temperature > 1000:
            import pdb

            pdb.set_trace()

        # Set the temperature of this layer appropriately.
        # @@@ This is probably not correct :p
        self.temperature = self.temperature + (
            excess_pv_heat
            - back_plate.conductance
            * self.area
            * (self.temperature - ambient_temperature)
        ) * resolution * 60 / (self._mass * self._heat_capacity)

        # The mass flow rate used is in kilograms per second. This requires multuplying
        # by the time for which the simulation was running to convert to the mass flow
        # rate
        self.output_water_temperature = input_water_temperature + _htf_heat_transfer(
            input_water_temperature,
            self.temperature,
            self.mass_flow_rate,
            self.htf_heat_capacity,
        ) * resolution * 60 / (
            self.mass_flow_rate * resolution * 60 * self.htf_heat_capacity
        )

        # ! At this point, it may be that the output water temperature from the panel
        # ! should be computed. We will see... :)
        self.temperature = (
            self.temperature
            - _htf_heat_transfer(
                input_water_temperature,
                self.temperature,
                self.mass_flow_rate,
                self.htf_heat_capacity,
            )
            / (self.htf_heat_capacity * self.mass_flow_rate * resolution * 60)
        )

        return self.output_water_temperature


class PVT:
    """
    Represents an entire PV-T collector.

    .. attribute:: timezone
        The timezone that the PVT system is based in.

    """

    # Private attributes:
    #
    # .. attribute:: _glass
    #   Represents the upper (glass) layer of the panel.
    #
    # .. attribute:: _air_gap_thickness
    #   The thickness of the air gap between the glass and PV (or collector) layers,
    #   measured in meters.
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
        glass_parameters: OpticalLayerParameters,
        collector_parameters: CollectorParameters,
        back_params: BackLayerParameters,
        air_gap_thickness: float,
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

        :param glass_parameters:
            Parameters used to instantiate the glass layer.

        :param collector_parameters:
            Parametsrs used to instantiate the collector layer.

        :param back_params:
            Parameters used to instantiate the back layer.

        :param air_gap_thickness:
            The thickness, in meters, of the air gap between the PV and glass layers.

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

        self._glass = Glass(glass_parameters)

        self._air_gap_thickness = air_gap_thickness

        if pv_layer_included and pv_parameters is not None:
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

    def _get_solar_irradiance(self, weather_conditions: WeatherConditions) -> float:
        """
        Returns the solar irradiance striking the panel normally.

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
            weather_conditions.irradiance * math.cos(math.radians(solar_diff_angle))
            if weather_conditions.declination >= MINIMUM_SOLAR_DECLINATION
            and 0 <= solar_diff_angle <= MAXIMUM_SOLAR_DIFF_ANGLE
            else 0
        )

    def update(
        self,
        input_water_temperature: float,
        resolution: float,
        weather_conditions: WeatherConditions,
    ) -> float:
        """
        Updates the properties of the PV-T collector based on a changed input temp..

        :param input_water_temperature:
            The water temperature going into the PV-T collector.

        :param resolution:
            The resolution of the model being run, measured in minutes.

        :param weather_conditions:
            The weather conditions at the time of day being incremented to.

        :return:
            The output water temperature from the PV-T panel.

        """

        solar_irradiance = (
            self._get_solar_irradiance(weather_conditions) * resolution * 60
        )

        # Call the pv panel to update its temperature.
        if self._pv is not None:
            # The excese PV heat generated is in units of Watts.
            excess_pv_heat = self._pv.update(
                solar_irradiance,
                input_water_temperature,
                weather_conditions.ambient_temperature,
                resolution,
                self._air_gap_thickness,
                self._glass.temperature,
                self._glass.emissivity,
                self._collector.temperature,
                self._collector.mass_flow_rate,
                self._collector.htf_heat_capacity,
                self._back_plate.conductance,
            )

            # Pass this new temperature through to the glass instance to update it.
            self._glass.update(
                self._air_gap_thickness,
                self._pv.area,
                self._pv.temperature,
                self._pv.emissivity,
                weather_conditions,
            )

            # Pass this new temperature through to the collector instance to update it.
            # The excess pv heat term is measured in Watts.
            output_water_temperature = self._collector.update(
                resolution,
                _pv_to_collector_conductive_transfer(
                    weather_conditions.ambient_temperature,
                    resolution,
                    input_water_temperature,
                    self._collector.temperature,
                    self._collector.mass_flow_rate,
                    self._collector.htf_heat_capacity,
                    self._back_plate.conductance,
                ),
                input_water_temperature,
                weather_conditions.ambient_temperature,
                self._back_plate,
            )

            return output_water_temperature
        raise Exception("Currently, a PV layer is needed.")

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
    def glass_temperature(self) -> float:
        """
        Returns the temperature of the glass layer of the PV-T system.

        :return:
            The temperature, in Kelvin, of the glass layer of the PV-T system.

        """

        return self._glass.temperature

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

    def electrical_output(
        self, resolution: float, weather_conditions: WeatherConditions
    ) -> float:
        """
        Returns the electrical output of the PV-T panel in Watts.

        :param resolution:
            The resolution - I think this is needed.
            @@@ Check this!

        :param weather_conditions:
            The current weather conditions at the time step being incremented to.

        :return:
            The electrical output, in Watts, of the PV-T panel.

        """

        return (
            self.electrical_efficiency
            * self._get_solar_irradiance(weather_conditions)
            * self._pv.area
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
