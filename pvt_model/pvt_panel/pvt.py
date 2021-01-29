#!/usr/bin/python3.7
########################################################################################
# pvt_panel/pvt.py - Models a PVT panel and all contained components.
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

from . import back_plate, collector, glass, pv

from ..__utils__ import (
    MissingParametersError,
    ProgrammerJudgementFault,
    BackLayerParameters,
    CollectorParameters,
    OpticalLayerParameters,
    PVParameters,
    WeatherConditions,
)
from .__utils__ import solar_heat_input

__all__ = ("PVT",)


# The minimum height in degrees that the sun must be above the horizon for light to
# strike the panel and be useful.
MINIMUM_SOLAR_DECLINATION = 5
# The maximum angle at which the panel can still be assumed to gain sunlight. This will
# limit both edge effects and prevent light reaching the panel from behind.
MAXIMUM_SOLAR_DIFF_ANGLE = 88


#####################
# PV-T Panel Layers #
#####################


class PVT:
    """
    Represents an entire PV-T collector.

    .. attribute:: timezone
        The timezone that the PVT system is based in.

    """

    # Private attributes:
    #
    # .. attribute:: _air_gap_thickness
    #   The thickness of the air gap between the glass and PV (or collector) layers,
    #   measured in meters.
    #
    # .. attribute:: _azimuthal_orientation
    #   The angle between the normal to the panel's surface and True North.
    #
    # .. attribute:: _collector
    #   Represents the lower (thermal-collector) layer of the panel.
    #
    # .. attribute:: _glass
    #   Represents the upper (glass) layer of the panel. This is set to `None` if the
    #   panel is unglazed.
    #
    # .. attribute:: _horizontal_tracking
    #   A `bool` giving whether or not the panel tracks horizontally.
    #
    # .. attribute:: _portion_covered
    #   The portion of the PV-T panel which is covered with PV.
    #
    # .. attribute:: _pv
    #   Represents the middle (pv) layer of the panel. Can be set to `None` if not
    #   present in the panel.
    #
    # .. attribute:: _pv_to_collector_thermal_conductance
    #   The thermal conductance, in Watts per meter squared Kelvin, between the PV layer
    #   and collector layer of the panel.
    #
    # .. attribute:: _tilt
    #   The angle between the normal to the panel's surface and the horizontal.
    #
    # .. attribute:: _vertical_tracking
    #   A `bool` giving whether or not the panel tracks verticallly.
    #

    def __init__(
        self,
        air_gap_thickness: float,
        area: float,
        back_params: BackLayerParameters,
        collector_parameters: CollectorParameters,
        glass_parameters: OpticalLayerParameters,
        glazed: bool,
        latitude: float,
        longitude: float,
        portion_covered: float,
        pv_parameters: Optional[PVParameters],
        pv_to_collector_thermal_conductance: float,
        timezone: datetime.timezone,
        *,
        azimuthal_orientation: Optional[float] = None,
        horizontal_tracking: bool = False,
        tilt: Optional[float] = None,
        vertical_tracking: bool = False,
    ) -> None:
        """
        Instantiate an instance of the PV-T collector class.

        :param air_gap_thickness:
            The thickness, in meters, of the air gap between the PV and glass layers.

        :param area:
            The area of the panel, measured in meters squared.

        :param back_params:
            Parameters used to instantiate the back layer.

        :param collector_parameters:
            Parametsrs used to instantiate the collector layer.

        :param glass_parameters:
            Parameters used to instantiate the glass layer.

        :param glazed:
            Whether the panel is glazed. I.E., whether the panel has a glass layer.

        :param latitude:
            The latitude of the PV-T system, defined in degrees relative to the equator
            in the standard way.

        :param longitude:
            The longitude of the PV-T system, defined in degrees relative to the
            Greenwich meridian.

        :param portion_covered:
            The portion of the PV-T panel which is covered with PV, ranging from 0
            (none) to 1 (all).

        :param pv_layer_included:
            Whether or not the system includes a photovoltaic layer.

        :param pv_parameters:
            Parameters used to instantiate the PV layer.

        :param azimuthal_orientation:
            The orientation of the surface of the panel, defined relative to True North,
            measured in degrees.

        :param horizontal_tracking:
            Whether or not the PV-T system tracks the sun horizontally.

        :param tilt:
            The tilt of the panel above the horizontal, measured in degrees.

        :param vertical_tracking:
            Whether or not the PV-T system tracks the sun vertically.

        """

        self._air_gap_thickness = air_gap_thickness
        self._azimuthal_orientation = azimuthal_orientation
        self._horizontal_tracking = horizontal_tracking
        self._latitude = latitude
        self._longitude = longitude
        self._portion_covered = portion_covered
        self._pv_to_collector_thermal_conductance = pv_to_collector_thermal_conductance
        self._vertical_tracking = vertical_tracking
        self._tilt = tilt
        self.area = area
        self.timezone = timezone

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

        # Instantiate the glass layer.
        if glazed:
            self._glass: Optional[glass.Glass] = glass.Glass(glass_parameters)
        else:
            self._glass = None

        # Instantiate the PV layer.
        if portion_covered != 0 and pv_parameters is not None:
            pv_parameters.area *= portion_covered
            self._pv: Optional[pv.PV] = pv.PV(pv_parameters)
        # If the PV layer parameters have not been specified, then raise an error.
        elif portion_covered != 0 and pv_parameters is None:
            raise MissingParametersError(
                "PVT",
                "PV-layer parameters must be provided if including a PV layer.",
            )
        else:
            self._pv = None

        # Instantiate the collector layer.
        self._collector = collector.Collector(collector_parameters)

        # Instantiate the back_plate layer.
        self._back_plate = back_plate.BackPlate(back_params)

    def __repr__(self) -> str:
        """
        A nice-looking representation of the PV-T panel.

        :return:
            A nice-looking representation of the PV-T panel.

        """

        return (
            "PVT("
            f"_back_plate: {self._back_plate}_"
            f"_collector: {self._collector}, "
            f"_glass: {self._glass}, "
            f"_pv: {self._pv}, "
            f"azimuthal_orientation: {self._azimuthal_orientation}, "
            f"coordinates: {self._latitude}N {self._longitude}E, "
            f"tilt: {self._tilt}deg, "
        )

    def _get_solar_orientation_diff(
        self, azimuthal_angle: float, declination: float
    ) -> float:
        """
        Determine the between the panel's normal and the sun.

        :param azimuthal_angle:
            The position of the sun, defined as degrees clockwise from True North.

        :param declination:
            The declination of the sun: the angle in degrees of the sun above the
            horizon.

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

    @property
    def bulk_water_temperature(self) -> float:
        """
        Returns the temperature, in Kelvin, of the "bulk water" HTF within the collector

        :return:
            The HTF temperature within the collector, measured in Kelvin.

        """

        return self._collector.bulk_water_temperature

    @property
    def collector_output_temperature(self) -> float:
        """
        Returns the output temperature, in Kelvin, of HTF from the collector.

        :return:
            The collector output temp in Kelvin.

        """

        return self._collector.output_water_temperature

    @property
    def collector_temperature(self) -> float:
        """
        Returns the temperature of the collector layer of the PV-T system.

        :return:
            The temperature, in Kelvin, of the collector layer of the PV-T system.

        """

        return self._collector.temperature

    @property
    def coordinates(self) -> Tuple[float, float]:
        """
        Returns a the coordinates of the panel.

        :return:
            A `tuple` containing the latitude and longitude of the panel.

        """

        return (self._latitude, self._longitude)

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

        return self._pv.electrical_efficiency

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

    def get_solar_irradiance(  # pylint: disable=no-self-use
        self, weather_conditions: WeatherConditions
    ) -> float:
        """
        Returns the irradiance striking the panel normally in Watts per meter squared.

        :param weather_conditions:
            The current weather conditions.

        :return:
            The solar irradiance striking the panel, adjusted for the sun's declination
            and azimuthal angle, as well as the panel's orientation, dependant on the
            panel's latitude and longitude.

        """

        # >>> Currently, Maria's profiles, rather than my Maths, are used.
        # # Compute the angle of solar irradiance wrt the panel.
        # solar_diff_angle = self._get_solar_orientation_diff(
        #     weather_conditions.azimuthal_angle,
        #     weather_conditions.declination,
        # )

        # return (
        #     weather_conditions.irradiance  # [W/m^2]
        #     * math.cos(math.radians(solar_diff_angle))
        #     if weather_conditions.declination >= MINIMUM_SOLAR_DECLINATION
        #     and 0 <= solar_diff_angle <= MAXIMUM_SOLAR_DIFF_ANGLE
        #     else 0
        # )
        # <<< End of my Maths.
        # >>> Beginning of Maria's profile fetching.
        return weather_conditions.irradiance
        # <<< End of Maria's profile fetching.

    @property
    def glass_temperature(self) -> Optional[float]:
        """
        Returns the temperature of the glass layer of the PV-T system.

        :return:
            The temperature, in Kelvin, of the glass layer of the PV-T system.

        """

        return self._glass.temperature if self._glass is not None else None

    @property
    def glazed(self) -> bool:
        """
        Returns whether the panel is glazed, ie whether it has a glass layer.

        :return:
            Whether the panel is glazed (True) or unglazed (False).

        """

        return self._glass is not None

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
    def mass_flow_rate(self) -> float:
        """
        Returns the mass flow rate through the collector, measured in kg/s.

        :return:
            The mass flow rate of heat-transfer fluid through the thermal collector.

        """

        return self._collector.mass_flow_rate

    @property
    def output_water_temperature(self) -> float:
        """
        Returns the temperature of the water outputted by the panel, measured in Kelvin.

        :return:
            The output water temperature from the thermal collector.

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

    def update(
        self,
        input_water_temperature: float,
        internal_resolution: float,
        weather_conditions: WeatherConditions,
    ) -> Tuple[float, float, float, float, Optional[float]]:
        """
        Updates the properties of the PV-T collector based on a changed input temp..

        :param input_water_temperature:
            The water temperature going into the PV-T collector.

        :param internal_resolution:
            The resolution of the model being run, measured in seconds.

        :param weather_conditions:
            The weather conditions at the time of day being incremented to.

        :return:
            A `tuple` containing:
            - the heat lost through the back plate, measured in Joules;
            - the heat gain by the bulk water, measured in Joules;
            - the output water temperature from the thermal collector, measured in
              Kelvin;
            - the upward heat lost from the collector layer, measured in Joules;
            - the upward heat lost from the glass layer, measured in Joules.

        """

        # Compute the solar energy inputted to the system in Joules per meter squared.
        solar_energy_input = (
            self.get_solar_irradiance(weather_conditions)  # [W/m^2]
            * internal_resolution  # [seconds]
        )  # [J/m^2]

        # Call the pv panel to update its temperature.
        pv_to_glass_heat_input: Optional[float] = None
        if self._pv is not None:
            # The excese PV heat generated is in units of Watts.
            collector_heat_input, pv_to_glass_heat_input = self._pv.update(
                air_gap_thickness=self._air_gap_thickness,
                collector_temperature=self._collector.temperature,
                glass_emissivity=self._glass.emissivity
                if self._glass is not None
                else None,
                glass_temperature=self._glass.temperature
                if self._glass is not None
                else None,
                glazed=self.glazed,
                internal_resolution=internal_resolution,
                pv_to_collector_thermal_conductance=self._pv_to_collector_thermal_conductance,
                solar_energy_input=solar_energy_input,
                weather_conditions=weather_conditions,
            )  # [J], [W]
            collector_heat_input += solar_heat_input(
                self._collector.absorptivity,
                self._collector.area * (1 - self._portion_covered),
                solar_energy_input,
                self._collector.transmissivity,
            )  # [J]
        else:
            collector_heat_input = solar_heat_input(
                self._collector.absorptivity,
                self._collector.area,
                solar_energy_input,
                self._collector.transmissivity,
            )  # [J]
            pv_to_glass_heat_input = None  # [W]

        # Based on the heat supplied, both from the sun (depending on whether there is
        # no PV layer present, or whether the PV layer does not fully cover the panel),
        # and from the heat transfered in from the PV layer.
        (
            back_plate_heat_loss,  # [J]
            bulk_water_heat_gain,  # [J]
            output_water_temperature,  # [K]
            collector_to_glass_heat_input,  # [J]
            upward_collector_heat_loss,  # [J]
        ) = self._collector.update(
            air_gap_thickness=self._air_gap_thickness,
            back_plate_instance=self._back_plate,
            collector_heat_input=collector_heat_input,
            glass_emissivity=self._glass.emissivity
            if self._glass is not None
            else None,
            glass_layer_included=self._glass is not None,
            glass_temperature=self._glass.temperature
            if self._glass is not None
            else None,
            input_water_temperature=input_water_temperature,
            internal_resolution=internal_resolution,
            portion_covered=self._portion_covered,
            weather_conditions=weather_conditions,
        )

        # Determine the heat inputted to the glass layer.
        if (
            self.glazed
            and pv_to_glass_heat_input is None
            and collector_to_glass_heat_input is None
        ):
            raise ProgrammerJudgementFault(
                "The panel has neither a PV or Collector layer if glazed."
            )
        glass_heat_input: float = 0  # [W]
        if pv_to_glass_heat_input is not None:
            glass_heat_input += pv_to_glass_heat_input
        if collector_to_glass_heat_input is not None:
            glass_heat_input += (
                collector_to_glass_heat_input / internal_resolution
            )  # [J]

        # Pass this new temperature through to the glass instance to update it.
        if self._glass is not None:
            upward_glass_heat_loss = self._glass.update(
                glass_heat_input, internal_resolution, weather_conditions
            )

            return (
                back_plate_heat_loss,  # [J]
                bulk_water_heat_gain,  # [J]
                output_water_temperature,  # [K]
                upward_collector_heat_loss,  # [J]
                upward_glass_heat_loss,  # [J]
            )

        return (
            back_plate_heat_loss,  # [J]
            bulk_water_heat_gain,  # [J]
            output_water_temperature,  # [K]
            upward_collector_heat_loss,  # [J]
            None,  # [J]
        )
