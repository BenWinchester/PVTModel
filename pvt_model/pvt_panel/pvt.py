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

    .. attribute:: air_gap_thickness
        The thickness of the air gap between the glass and PV (or collector) layers,
        measured in meters.

    .. attribute:: collector
      Represents the lower (thermal-collector) layer of the panel.

    .. attribute:: glass
      Represents the upper (glass) layer of the panel. This is set to `None` if the
      panel is unglazed.

    .. attribute:: latitude
        The latitude of the panel, measured in degrees.

    .. attribute:: longitude
        The longitude of the panel, measured in degrees.

    .. attribute:: portion_covered
        The portion of the PV-T panel which is covered with PV.

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
    # .. attribute:: pv
    #   Represents the middle (pv) layer of the panel. Can be set to `None` if not
    #   present in the panel.
    #
    # .. attribute:: pv_to_collector_thermal_conductance
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
        diffuse_reflection_coefficient: float,
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

        :param diffuse_reflection_coefficient:
            The coefficient of diffuse reflectivity of the upper layer.

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

        :raises: MissingParametesrError
            Raised if not all the parameters needed are supplied.

        """

        self.air_gap_thickness = air_gap_thickness
        self._azimuthal_orientation = azimuthal_orientation
        self._horizontal_tracking = horizontal_tracking
        self.latitude = latitude
        self.longitude = longitude
        self.portion_covered = portion_covered
        self.pv_to_collector_thermal_conductance = pv_to_collector_thermal_conductance
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
            self.glass: glass.Glass = glass.Glass(
                diffuse_reflection_coefficient, glass_parameters
            )
        else:
            raise ProgrammerJudgementFault(
                "A glass layer is required in the current set up."
            )

        # Instantiate the PV layer.
        if portion_covered != 0 and pv_parameters is not None:
            pv_parameters.area *= portion_covered
            self.pv: pv.PV = pv.PV(pv_parameters)
        # If the PV layer parameters have not been specified, then raise an error.
        elif portion_covered != 0 and pv_parameters is None:
            raise MissingParametersError(
                "PVT",
                "PV-layer parameters must be provided if including a PV layer.",
            )
        else:
            raise ProgrammerJudgementFault(
                "A PV layer is required in the current set up."
            )

        # Instantiate the collector layer.
        self.collector = collector.Collector(collector_parameters)

        # Instantiate the back_plate layer.
        self.back_plate = back_plate.BackPlate(back_params)

    def __repr__(self) -> str:
        """
        A nice-looking representation of the PV-T panel.

        :return:
            A nice-looking representation of the PV-T panel.

        """

        return (
            "PVT("
            f"back_plate: {self.back_plate}_"
            f"collector: {self.collector}, "
            f"glass: {self.glass}, "
            f"pv: {self.pv}, "
            f"azimuthal_orientation: {self._azimuthal_orientation}, "
            f"coordinates: {self.latitude}N {self.longitude}E, "
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
