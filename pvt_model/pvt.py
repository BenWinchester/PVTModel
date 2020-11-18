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

from .__utils__ import *

__all__ = ("PVT",)


# The minimum height in degrees that the sun must be above the horizon for light to
# strike the panel and be useful.
MINIMUM_SOLAR_DECLINATION = 5


##############
# Exceptions #
##############


class MissingParametersError(Exception):
    """
    Raised when not all parameters have been specified that are needed to instantiate.

    """

    def __init__(self, class_name, message) -> None:
        """
        Instantiate a missing parameters error.

        :param class_name:
            The class for which insufficient parameters were specified.

        :param message:
            An appended message to display to the user.

        """

        super().__init__(
            f"Missing parameters when initialising a '{class_name}' class: {message}."
        )


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
        The solar irradiance, normal to the panel, measured in Watts per meter
        squared.

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
        The solar heating input, measured in Watts.

    """

    return (
        (pv_transmissivity * pv_absorptivity)
        * solar_irradiance
        * pv_area
        * (1 - pv_efficiency)
    )


def _htf_heat_transfer(
    input_water_temperature: float,
    output_water_temperature: float,
    collector_mass_flow_rate: float,
) -> float:
    """
    Computes the heat transfer to the heat-transfer fluid.

    :param input_water_temperature:
        The input water temperature to the PV-T panel, measured in Kelvin.

    :param output_water_temperature:
        The output water temperature from the PV-T panel, measured in Kelvin.

    :param collector_mass_flow_rate:
        The mass-flow rate of heat-transfer fluid through the collector.

    :return:
        The heat transfer, in Watts, to the heat transfer fluid.

    """

    return (
        collector_mass_flow_rate
        * HEAT_CAPACITY_OF_WATER
        * (output_water_temperature - input_water_temperature)
    )


def _pv_to_collector_conductive_transfer(
    ambient_temperature: float,
    input_water_temperature: float,
    output_water_temperature: float,
    collector_temperature: float,
    collector_mass_flow_rate: float,
    back_plate_conductance: float,
) -> float:
    """
    Computes the heat transfer from the PV layer to the thermal collector.

    :param ambient_temperature:
        The ambient temperature in Kelvin.

    :param input_water_temperature:
        The input water temperature to the PV-T panel, measured in Kelvin.

    :param output_water_temperature:
        The output water temperature from the PV-T panel, measured in Kelvin.

    :param collector_temperature:
        The temperature of the collector layer, measured in Kelvin.

    :param collector_mass_flow_rate:
        The mass-flow rate of heat-transfer fluid through the collector.

    :param back_plate_conductance:
        The conductance of the back plate of the panel.

    :return:
        The heat transfer, in Watts, from the PV layer to the thermal-collector
        layer.

    """

    back_plate_heat_loss = back_plate_conductance * (
        collector_temperature - ambient_temperature
    )

    htf_heat_transfer = _htf_heat_transfer(
        input_water_temperature, output_water_temperature, collector_mass_flow_rate
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
        THERMAL_CONDUCTIVITY_OF_AIR,
        *(pv_temperature - glass_temp) * pv_area / air_gap_thickness,
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
                back_params.mass, back_params.heat_capacity, back_params.area
            )
        )

        self.conductivity = back_params.conductivity

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


class Collector(_Layer):
    """
    Represents the thermal collector (lower) layer of the PV-T panel.

    .. attribute:: output_water_temperature
        The temperature of the water outputted by the layer, measured in Kelvin.

    """

    def __init__(self, collector_params: CollectorParameters) -> None:
        """
        Instantiate a collector layer.

        :param output_water_temperature:
            The temperature of the water outputted by the layer, measured in Kelvin.

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
        self.mass_flow_rate = collector_params.mass_flow_rate

    def update(
        self,
        excess_pv_heat: float,
        input_water_temperature: float,
        ambient_temperature: float,
        back_plate: BackPlate,
    ) -> float:
        """
        Update the internal properties of the PV layer based on external factors.

        :param excess_pv_heat:
            Excess heat from the adjoining PV layer, measured in who-knows what.

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

        # Set the temperature of this layer appropriately.
        # @@@ This is probably not correct :p
        self.temperature = self.temperature + excess_pv_heat / (
            self._mass * self._heat_capacity
        )

        self.output_water_temperature = input_water_temperature + (
            1 / (self.mass_flow_rate * HEAT_CAPACITY_OF_WATER)
        ) * (
            excess_pv_heat
            - back_plate.conductance * (self.temperature - ambient_temperature)
        )

        # ! At this point, it may be that the output water temperature from the panel
        # ! should be computed. We will see... :)
        return self.output_water_temperature


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

    def excess_heat(
        self,
        solar_irradiance: float,
        input_water_temperature: float,
        output_water_temperature: float,
        ambient_temperature: float,
        air_gap_thickness: float,
        glass_temp: float,
        glass_emissivity: float,
        collector_temperature: float,
        collector_mass_flow_rate: float,
        back_plate_conductance: float,
    ) -> float:
        """
        Computes the excess heat provided to the PV layer.

        :param solar_irradiance:
            The solar irradiance, normal to the panel, measured in Watts per meter
            squared.

        :param input_water_temperature:
            The input water temperature to the panel, measured in Kelvin.

        :param output_water_temperature:
            The output water temperature from the panel, measured in Kelvin.

        :param ambient_temperature:
            The ambient temperature of the air surrounding the panel, measured in
            Kelvin.

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

        :param back_plate_conductance:
            The conductante of the back plate of the PV-T panel.

        :return:
            The excess heat after balancing the heat inputs and outputs for the layer.

        """

        return _solar_heat_input(
            solar_irradiance,
            self.area,
            self.efficiency,
            self._absorptivity,
            self._transmissivity,
        ) - self.area * (
            pv_to_glass_radiative_transfer(
                glass_temp,
                glass_emissivity,
                self.area,
                self.temperature,
                self.emissivity,
            )
            + pv_to_glass_conductive_transfer(
                air_gap_thickness, glass_temp, self.area, self.temperature
            )
            + _pv_to_collector_conductive_transfer(
                ambient_temperature,
                input_water_temperature,
                output_water_temperature,
                collector_temperature,
                collector_mass_flow_rate,
                back_plate_conductance,
            )
        )

    def update(
        self,
        solar_irradiance: float,
        input_water_temperature: float,
        output_water_temperature: float,
        ambient_temperature: float,
        air_gap_thickness: float,
        glass_temp: float,
        glass_emissivity: float,
        collector_temperature: float,
        collector_mass_flow_rate: float,
        back_plate_conductance: float,
    ) -> float:
        """
        Update the internal properties of the PV layer based on external factors.

        :param solar_irradiance:
            The solar irradiance, normal to the panel, measured in Watts per meter
            squared.

        :param input_water_temperature:
            The input water temperature to the panel, measured in Kelvin.

        :param output_water_temperature:
            The output water temperature of the panel, measured in Kelvin.

        :param ambient_temperature:
            The ambient temperature of the air surrounding the panel, measured in
            Kelvin.

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

        :param back_plate_conductance:
            The conductance of the back-plate of the PV-T panel.

        :return:
            The excess heat provided which is transferred to the thermal collector.

        """

        # Determine the excess heat that has been inputted into the panel.
        excess_heat = self.excess_heat(
            solar_irradiance,
            input_water_temperature,
            output_water_temperature,
            ambient_temperature,
            air_gap_thickness,
            glass_temp,
            glass_emissivity,
            collector_temperature,
            collector_mass_flow_rate,
            back_plate_conductance,
        )

        # Use this to compute the rise in temperature of the PV layer and set the
        # temperature appropriately.
        self.temperature = self.temperature + excess_heat / (
            self._mass * self._heat_capacity
        )

        # Return the excess heat, transferred to the collector, based on the new PV-T
        # temperature.
        return self.excess_heat(
            solar_irradiance,
            input_water_temperature,
            output_water_temperature,
            ambient_temperature,
            air_gap_thickness,
            glass_temp,
            glass_emissivity,
            collector_temperature,
            collector_mass_flow_rate,
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
            1 - self._thermal_coefficient * (self.temperature)
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

    def _layer_to_air_conductive_transfer(
        self, ambient_temperature: float, wind_speed: float
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
        weather_conditions: WeatherConditions,
        pv_layer: PV,
    ) -> None:
        """
        Update the internal properties of the PV layer based on external factors.

        :param air_gap_thickness:
            The thickness, in meters, of the air gap between the PV and glass layers.

        :param weather_conditions:
            The weather conditions at the current time step.

        :param pv_layer:
            The PV layer of the PV-T panel.

        """

        # Set the temperature of this layer appropriately.
        heat_losses = self._layer_to_air_conductive_transfer(
            weather_conditions.ambient_temperature, weather_conditions.wind_speed
        ) + self._layer_to_sky_radiative_transfer(weather_conditions.sky_temperature)

        heat_input = pv_layer.pv_to_glass_conductive_transfer(
            air_gap_thickness, self.temperature
        ) + pv_layer.pv_to_glass_radiative_transfer(self.temperature, self.emissivity)

        excess_heat = heat_input - heat_losses

        self.temperature = self.temperature + excess_heat / (
            self._mass * self._heat_capacity
        )


class PVT:
    """
    Represents an entire PV-T collector.

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
        glass_parameters: LayerParameters,
        collector_parameters: LayerParameters,
        back_params: BackLayerParameters,
        air_gap_thickness: float,
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

        if pv_layer_included:
            self._pv = PV(pv_parameters)
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

        if self._horizontal_tracking:
            horizontal_diff: float = 0
        else:
            horizontal_diff = abs(self._azimuthal_orientation - azimuthal_angle)

        if self._vertical_tracking:
            vertical_diff: float = 0
        else:
            vertical_diff = abs(self._tilt - declination)

        # Combine these to generate the angle between the two directions.
        return math.acos(math.cos(horizontal_diff) * math.cos(vertical_diff))

    def update(
        self,
        input_water_temperature: float,
        weather_conditions: WeatherConditions,
    ) -> float:
        """
        Updates the properties of the PV-T collector based on a changed input temp..

        :param input_water_temperature:
            The water temperature going into the PV-T collector.

        :param weather_conditions:
            The weather conditions at the time of day being incremented to.

        :return:
            The output water temperature from the PV-T panel.

        """

        # Compute the angle of solar irradiance wrt the panel.
        solar_diff_angle = self._get_solar_orientation_diff(
            weather_conditions.declination, weather_conditions.azimuthal_angle
        )
        solar_irradiance = (
            weather_conditions.irradiance * math.cos(solar_diff_angle)
            if weather_conditions.declination >= MINIMUM_SOLAR_DECLINATION
            else 0
        )

        # Call the pv panel to update its temperature.
        if self._pv is not None:
            excess_pv_heat = self._pv.update(
                solar_irradiance,
                input_water_temperature,
                self._collector.output_water_temperature,
                weather_conditions.ambient_temperature,
                self._air_gap_thickness,
                self._glass.temperature,
                self._glass.emissivity,
                self._collector.temperature,
                self._collector.mass_flow_rate,
                self._back_plate.conductance,
            )

            # Pass this new temperature through to the glass instance to update it.
            self._glass.update(self._air_gap_thickness, weather_conditions, self._pv)

            # Pass this new temperature through to the collector instance to update it.
            output_water_temperature = self._collector.update(
                excess_pv_heat,
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
    def output_water_temperature(self) -> float:
        """
        Returns the temperature of the water outputted by the panel, measured in Kelvin.

        :return:
            The output water temperature from the thermal collector.

        """

        return self._collector.output_water_temperature
