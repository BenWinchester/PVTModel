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

from typing import Dict, Optional, Tuple


from . import bond, collector, glass, pv

from ...__utils__ import MissingParametersError

from ..__utils__ import (
    CollectorParameters,
    OpticalLayerParameters,
    PVParameters,
    WeatherConditions,
)
from ..constants import (
    FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR,
    THERMAL_CONDUCTIVITY_OF_AIR,
)

from .__utils__ import MicroLayer
from .segment import Segment, SegmentCoordinates

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

    .. attribute:: adhesive
        Represents the adhesive layer between the pv layer and the thermal absorber
        layer.

    .. attribute:: air_gap_thickness
        The thickness of the air gap between the glass and PV (or collector) layers,
        measured in meters.

    .. attribute:: bond
        Represents the bond between the absorber (collector) layer and the riser tube
        pipes.

    .. attribute:: collector
        Represents the lower (thermal-collector) layer of the panel.

    .. attribute:: eva
        Represents the EVA layer between the PV panel and the collector layer.

    .. attribute:: glass
        Represents the upper (glass) layer of the panel. This is set to `None` if the
        panel is unglazed.

    .. attribute:: insulation
        Represents the thermal insulation on the back of the PV-T collector.

    .. attribute:: latitude
        The latitude of the panel, measured in degrees.

    .. attribute:: longitude
        The longitude of the panel, measured in degrees.

    .. attribute:: portion_covered
        The portion of the PV-T panel which is covered with PV.

    .. attribute:: pv
         Represents the middle (pv) layer of the panel. Can be set to `None` if not
        present in the panel.

    .. attribute:: pv_to_collector_thermal_conductance
        The thermal conductance, in Watts per meter squared Kelvin, between the PV layer
        and collector layer of the panel.

    .. attribute:: segments
        A mapping between segment coordinates and the segment.

    .. attribute:: tedlar
        Represents the tedlar back plate to the PV layer, situated above the PV layer.

    .. attribute:: timezone
        The timezone that the PVT system is based in.

    """

    # Private attributes:
    #
    # .. attribute:: _azimuthal_orientation
    #   The angle between the normal to the panel's surface and True North.
    #
    # .. attribute:: _horizontal_tracking
    #   A `bool` giving whether or not the panel tracks horizontally.
    #
    # .. attribute:: _tilt
    #   The angle between the normal to the panel's surface and the horizontal.
    #
    # .. attribute:: _vertical_tracking
    #   A `bool` giving whether or not the panel tracks verticallly.
    #

    def __init__(
        self,
        absorber_pipe_bond: bond.Bond,
        adhesive: MicroLayer,
        air_gap_thickness: float,
        area: float,
        collector_parameters: CollectorParameters,
        diffuse_reflection_coefficient: float,
        eva: MicroLayer,
        glass_parameters: OpticalLayerParameters,
        insulation: MicroLayer,
        latitude: float,
        length: float,
        longitude: float,
        portion_covered: float,
        pv_parameters: PVParameters,
        pv_to_collector_thermal_conductance: float,
        segments: Dict[SegmentCoordinates, Segment],
        tedlar: MicroLayer,
        timezone: datetime.timezone,
        width: float,
        *,
        azimuthal_orientation: Optional[float] = None,
        horizontal_tracking: bool = False,
        tilt: Optional[float] = None,
        vertical_tracking: bool = False,
    ) -> None:
        """
        Instantiate an instance of the PV-T collector class.

        :param adhesive:
            A :class:`MicroLayer` instance representing the adhesive layer.

        :param air_gap_thickness:
            The thickness, in meters, of the air gap between the PV and glass layers.

        :param area:
            The area of the panel, measured in meters squared.

        :param bond:
            A :class:`Bond` instance representing the bond layer.

        :param collector_parameters:
            Parametsrs used to instantiate the collector layer.

        :param diffuse_reflection_coefficient:
            The coefficient of diffuse reflectivity of the upper layer.

        :param eva:
            A :class:`MicroLayer` instance representing the eva layer.

        :param glass_parameters:
            Parameters used to instantiate the glass layer.

        :param insulation:
            A :class:`MicroLayer` instance representing the insulation on the back of
            the PV-T collector.

        :param latitude:
            The latitude of the PV-T system, defined in degrees relative to the equator
            in the standard way.

        :param length:
            The length of the collector in meters.

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

        :param segments:
            A mapping between segment coordinate and segment for all segments to be
            included in the layer.

        :param tedlar:
            A :class:`MicroLayer` instance representing the tedlar layer.

        :param timezone:
            The timezone in which the PV-T system is installed.

        :param width:
            The width of the collector in meters.

        :param azimuthal_orientation:
            The orientation of the surface of the panel, defined relative to True North,
            measured in degrees.

        :param horizontal_tracking:
            Whether or not the PV-T system tracks the sun horizontally.

        :param tilt:
            The tilt of the panel above the horizontal, measured in degrees.

        :param vertical_tracking:
            Whether or not the PV-T system tracks the sun vertically.

        :raises: MissingParametesrError
            Raised if not all the parameters needed are supplied.

        """

        self._azimuthal_orientation = azimuthal_orientation
        self._horizontal_tracking = horizontal_tracking
        self._tilt = tilt
        self._vertical_tracking = vertical_tracking
        self.air_gap_thickness = air_gap_thickness
        self.area = area
        self.latitude = latitude
        self.length = length
        self.longitude = longitude
        self.portion_covered = portion_covered
        self.pv_to_collector_thermal_conductance = pv_to_collector_thermal_conductance
        self.segments = segments
        self.timezone = timezone
        self.width = width

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

        # Instantiate the layers.
        self.adhesive = adhesive
        self.bond = absorber_pipe_bond
        self.collector = collector.Collector(collector_parameters)
        self.eva = eva
        self.glass: glass.Glass = glass.Glass(
            diffuse_reflection_coefficient, glass_parameters
        )
        self.insulation = insulation
        self.pv: pv.PV = pv.PV(pv_parameters)
        self.tedlar = tedlar

        # * Instantiate and store the segments on the class.

    def __repr__(self) -> str:
        """
        A nice-looking representation of the PV-T panel.

        :return:
            A nice-looking representation of the PV-T panel.

        """

        return (
            "PVT(\n"
            f"  glass: {self.glass},\n"
            f"  pv: {self.pv},\n"
            f"  eva: {self.eva},\n"
            f"  adhesive: {self.adhesive}\n"
            f"  tedlar: {self.tedlar}\n"
            f"  collector: {self.collector},\n"
            f"  bond: {self.bond},\n"
            f"  azimuthal_orientation: {self._azimuthal_orientation}, "
            f"coordinates: {self.latitude}N {self.longitude}E, "
            f"tilt: {self._tilt}deg"
            ")"
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
    def air_gap_heat_transfer_coefficient(self) -> float:
        """
        Gives the heat-transfer coefficient for heat transfer across the air gap.

        :return:
            The heat transfer coefficient across the air gap, measured in Watts per
            meter Kelvin.

        """

        return (
            THERMAL_CONDUCTIVITY_OF_AIR
            / self.air_gap_thickness
            * (
                1
                # @@@ FIXME - Additional code needed here to more accurately match
                # Ilaria's model. See page 75 of the equation specification.
            )
        )

    @property
    def air_gap_resistance(self) -> float:
        """
        Returns the thermal resistance of the air gap between the PV and glass layers.

        :return:
            The thermal resistance, measured in Kelvin meter squared per Watt.

        """

        return (
            self.eva.thickness / self.eva.conductivity
            + self.glass.thickness / self.glass.conductivity
            + self.pv.thickness / (2 * self.pv.conductivity)
            + self.glass.thickness / (2 * self.glass.conductivity)
            + 1 / self.air_gap_heat_transfer_coefficient
        )

    @property
    def coordinates(self) -> Tuple[float, float]:
        """
        Returns a the coordinates of the panel.

        :return:
            A `tuple` containing the latitude and longitude of the panel.

        """

        return (self.latitude, self.longitude)

    def electrical_output(
        self, pv_temperature: float, weather_conditions: WeatherConditions
    ) -> float:
        """
        Returns the electrical output of the PV-T panel in Watts.

        NOTE: We here need to include the portion of the panel that is covered s.t. the
        correct electricitiy-generating area is accounted for, rather than accidentailly
        inculding areas which do not generated electricity.

        :param pv_temperature:
            The temperature of the PV layer, measured in Kelvin.

        :param weather_conditions:
            The current weather conditions at the time step being incremented to.

        :return:
            The electrical output, in Watts, of the PV-T panel.

        """

        if self.pv is not None:
            electrical_output: float = (
                self.pv.electrical_efficiency(pv_temperature)
                * self.get_solar_irradiance(weather_conditions)
                * self.area
                * self.portion_covered
            )
        else:
            electrical_output = 0

        return electrical_output

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
    def insulation_thermal_resistance(self) -> float:
        """
        Returns the thermal resistance between the back layer of the collector and air.

        Insulation on the back of the PV-T collector causes there to be some thermal
        resistance to the heat transfer out of the back of the thermal collector. This
        value is computed here and returned.

        :return:
            The thermal resistance between the back layer of the collector and the
            surrounding air, measured in meter squared Kelvin per Watt.

        """

        return (
            self.insulation.thickness / self.insulation.conductivity
            + 1 / FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR
        )

    @property
    def pv_to_collector_thermal_resistance(self) -> float:
        """
        Returns the thermal resistance between the PV and collector layers.

        :return:
            The thermal resistance, in meters squared Kelvin per Watt, between the PV
            and collector layers.

        """

        return (
            self.eva.thickness / self.eva.conductivity
            + self.tedlar.thickness / self.tedlar.conductivity
            + self.adhesive.thickness / self.adhesive.conductivity
        )
