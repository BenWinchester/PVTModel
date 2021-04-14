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
panel, and thermal absorber. The model works by incrimenting the parameters through at
increasing time steps, and the code here aims to emaulate this.

"""

import datetime
import math

from typing import Dict, Optional, Tuple

import numpy

from . import bond, absorber, glass, pv

from ...__utils__ import MissingParametersError

from ..__utils__ import (
    CollectorParameters,
    OpticalLayerParameters,
    PVParameters,
    WeatherConditions,
)
from ..constants import (
    FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR,
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


####################
# Helper functions #
####################


def _conductive_heat_transfer_coefficient_with_gap(
    air_gap_thickness: float, weather_conditions: WeatherConditions
) -> float:
    """
    Computes the conductive heat transfer between the two layers, measured in W/m^2*K.

    The value computed is positive if the heat transfer is from the source to the
    destination, as determined by the arguments, and negative if the flow of heat is
    the reverse of what is implied via the parameters.

    The value for the heat transfer is returned in Watts.

    :param air_gap_thickness:
        The thickness of the air gap between the PV and glass layers.

    :param weather_conditions:
        The weather conditions at the time step being investigated.

    :return:
        The heat transfer coefficient, in Watts per meter squared Kelvin, between the
        two layers.

    """

    return (
        weather_conditions.thermal_conductivity_of_air / air_gap_thickness
    )  # [W/m*K] / [m]


#####################
# PV-T Panel Layers #
#####################


class PVT:
    """
    Represents an entire PV-T absorber.

    .. attribute:: adhesive
        Represents the adhesive layer between the pv layer and the thermal absorber
        layer.

    .. attribute:: air_gap_thickness
        The thickness of the air gap between the glass and PV (or absorber) layers,
        measured in meters.

    .. attribute:: bond
        Represents the bond between the absorber (absorber) layer and the riser tube
        pipes.

    .. attribute:: absorber
        Represents the lower (thermal-absorber) layer of the panel.

    .. attribute:: eva
        Represents the EVA layer between the PV panel and the absorber layer.

    .. attribute:: glass
        Represents the upper (glass) layer of the panel. This is set to `None` if the
        panel is unglazed.

    .. attribute:: insulation
        Represents the thermal insulation on the back of the PV-T absorber.

    .. attribute:: latitude
        The latitude of the panel, measured in degrees.

    .. attribute:: longitude
        The longitude of the panel, measured in degrees.

    .. attribute:: portion_covered
        The portion of the PV-T panel which is covered with PV.

    .. attribute:: pv
         Represents the middle (pv) layer of the panel. Can be set to `None` if not
        present in the panel.

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
        absorber_parameters: CollectorParameters,
        diffuse_reflection_coefficient: float,
        eva: MicroLayer,
        glass_parameters: OpticalLayerParameters,
        insulation: MicroLayer,
        latitude: float,
        length: float,
        longitude: float,
        portion_covered: float,
        pv_parameters: PVParameters,
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
        Instantiate an instance of the PV-T absorber class.

        :param adhesive:
            A :class:`MicroLayer` instance representing the adhesive layer.

        :param air_gap_thickness:
            The thickness, in meters, of the air gap between the PV and glass layers.

        :param area:
            The area of the panel, measured in meters squared.

        :param bond:
            A :class:`Bond` instance representing the bond layer.

        :param absorber_parameters:
            Parametsrs used to instantiate the absorber layer.

        :param diffuse_reflection_coefficient:
            The coefficient of diffuse reflectivity of the upper layer.

        :param eva:
            A :class:`MicroLayer` instance representing the eva layer.

        :param glass_parameters:
            Parameters used to instantiate the glass layer.

        :param insulation:
            A :class:`MicroLayer` instance representing the insulation on the back of
            the PV-T absorber.

        :param latitude:
            The latitude of the PV-T system, defined in degrees relative to the equator
            in the standard way.

        :param length:
            The length of the absorber in meters.

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
            The width of the absorber in meters.

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
        self.absorber = absorber.Collector(absorber_parameters)
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
            f"  absorber: {self.absorber},\n"
            f"  bond: {self.bond},\n"
            f"  outer_pipe_diameter: {self.absorber.outer_pipe_diameter}m\n"
            "  additional_params: "
            f"azimuthal_orientation: {self._azimuthal_orientation}, "
            f"coordinates: {self.latitude}N {self.longitude}E, "
            f"tilt: {self._tilt}deg"
            "\n)"
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

    def air_gap_resistance(self, weather_conditions: WeatherConditions) -> float:
        """
        Returns the thermal resistance of the air gap between the PV and glass layers.

        :param weather_conditions:
            The weather conditions at the time step being investigated.

        :return:
            The thermal resistance, measured in Kelvin meter squared per Watt.

        """

        return (
            self.eva.thickness / self.eva.conductivity
            + self.glass.thickness / self.glass.conductivity
            + self.pv.thickness / (2 * self.pv.conductivity)
            + self.glass.thickness / (2 * self.glass.conductivity)
            + 1
            / _conductive_heat_transfer_coefficient_with_gap(
                self.air_gap_thickness, weather_conditions
            )
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
    def glass_transmissivity_absorptivity_product(self) -> float:
        """
        Returns the transmissivity-absorptivity product of the glass layer.

        Due to internal reflections etc., an estimate is needed for the overall fraction
        of incident solar light that is absorbed by the glass layer.

        :return:
            The TA product of the glass layer.

        """

        ta_product: float = (
            1
            - self.glass.reflectance
            - self.glass.transmittance
            * (1 - self.pv.reflectance + self.pv.reflectance * self.glass.transmittance)
            / (1 - self.pv.reflectance * self.glass.reflectance)
        )

        return ta_product

    @property
    def insulation_thermal_resistance(self) -> float:
        """
        Returns the thermal resistance between the back layer of the absorber and air.

        Insulation on the back of the PV-T absorber causes there to be some thermal
        resistance to the heat transfer out of the back of the thermal absorber. This
        value is computed here and returned.

        :return:
            The thermal resistance between the back layer of the absorber and the
            surrounding air, measured in meter squared Kelvin per Watt.

        """

        return (
            self.insulation.thickness / self.insulation.conductivity
            + 1 / FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR
        )

    @property
    def pv_to_absorber_thermal_resistance(self) -> float:
        """
        Returns the thermal resistance between the PV and absorber layers.

        :return:
            The thermal resistance, in meters squared Kelvin per Watt, between the PV
            and absorber layers.

        """

        return (
            self.eva.thickness / self.eva.conductivity
            + self.tedlar.thickness / self.tedlar.conductivity
            + self.adhesive.thickness / self.adhesive.conductivity
        )

    @property
    def pv_transmissivity_absorptivity_product(self) -> float:
        """
        Returns the transmissivity-absorptivity product of the PV layer.

        Due to internal reflections etc., an estimate is needed for the overall fraction
        of incident solar light that is absorbed by the PV layer.

        :return:
            The TA product of the PV layer.

        """

        ta_product: float = (
            (1 - self.pv.reflectance - self.pv.transmittance) * self.glass.transmittance
        ) / (1 - self.pv.reflectance * self.glass.reflectance)

        return ta_product

    @property
    def tilt_in_radians(self) -> float:
        """
        Returns the tilt of the panel from the horizontal, measured in radians.

        :return:
            The angle between the plane of the panel and the horizontal, measured in
            radians.

        """

        return numpy.pi * self._tilt / 180
