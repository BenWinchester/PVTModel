#!/usr/bin/python3.7
########################################################################################
# pvt_panel/test_pv.py - MUT for the PV Tmodule.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Module Unit Tests (MUT) for the PVT module.

This module tests the various internal and external methods provided by the PVT panel.

"""

import math
import unittest

from datetime import timedelta, timezone
from typing import Optional
from unittest import mock  # pylint: disable=unused-import

from ...__utils__ import (
    BackLayerParameters,
    CollectorParameters,
    MissingParametersError,
    OpticalLayerParameters,
    ProgrammerJudgementFault,
    PVParameters,
    WeatherConditions,
)
from .. import pvt


class _BaseTest(unittest.TestCase):
    """
    Contains common functionality and helper functions for the test suite.

    """

    def _pvt(
        self,
        azimuthal_orientation: Optional[float] = 180,
        horizontal_tracking: bool = False,
        glazed: bool = True,
        latitude: float = 53.55,
        longitude: float = 0.011,
        tilt: Optional[float] = 35,
        portion_covered: float = 0.75,
        pv_params_provded: bool = True,
        vertical_tracking: bool = False,
    ) -> pvt.PVT:
        """
        Instantiate a :class:`pvt.PVT` instance based on parameters passed in.

        :param azimuthal_orientation:
            The azimuthal orientation of the panel: 0 for due North, 180 for due South.

        :param horizontal_tracking:
            Whether or not the panel tracks horizontally.

        :param glazed:
            Whether or not the panel has a glass layer.

        :param tilt:
            The tilt of the panel, in degrees, above the horzion.

        :param portion_covered:
            The portion of the panel which is covered with photovoltaics.

        :param pv_params_provided:
            Whether PV parameters should be provided to the :class:`pvt.PVT` instantiate
            method.

        :param vertical_tracking:
            Whether or not the panel tracks vertically.

        :return:
            A :class:`pvt.PVT` instance based on the parameters passed in.

        """

        back_layer_parameters = BackLayerParameters(
            mass=100, heat_capacity=2500, area=15, thickness=0.15, conductivity=500
        )
        collector_parameters = CollectorParameters(
            mass=100,
            heat_capacity=4000,
            area=15,
            thickness=0.05,
            transmissivity=0.9,
            absorptivity=0.88,
            emissivity=0.3,
            bulk_water_temperature=300,
            htf_heat_capacity=4180,
            length=1,
            mass_flow_rate=108,
            number_of_pipes=11,
            output_water_temperature=373,
            pipe_diameter=0.05,
        )
        glass_parameters = OpticalLayerParameters(
            mass=150,
            heat_capacity=4000,
            area=100,
            thickness=0.015,
            transmissivity=0.9,
            absorptivity=0.88,
            emissivity=0.5,
        )
        pv_parameters = PVParameters(
            mass=150,
            heat_capacity=4000,
            area=100,
            thickness=0.015,
            transmissivity=0.9,
            absorptivity=0.88,
            emissivity=0.5,
            reference_efficiency=0.15,
            reference_temperature=300,
            thermal_coefficient=0.1,
        )

        return pvt.PVT(
            air_gap_thickness=0.05,
            area=15,
            back_params=back_layer_parameters,
            collector_parameters=collector_parameters,
            diffuse_reflection_coefficient=0.18,
            glass_parameters=glass_parameters,
            glazed=glazed,
            latitude=latitude,
            longitude=longitude,
            portion_covered=portion_covered,
            pv_parameters=pv_parameters if pv_params_provded else None,
            pv_to_collector_thermal_conductance=500,
            timezone=timezone(timedelta(0)),
            azimuthal_orientation=azimuthal_orientation,
            horizontal_tracking=horizontal_tracking,
            tilt=tilt,
            vertical_tracking=vertical_tracking,
        )


class TestInstantiate(_BaseTest):
    """
    Tests the instantisation of the :class:`pvt.PVT` instance, probing all paths.

    """

    def test_mainline(self) -> None:
        """
        Tests the mainline case for PVT class instantisation.

        """

        try:
            self._pvt()
        except MissingParametersError:
            self.fail("Exception raised in the mainline case.")

    def test_vertical_error(self) -> None:
        """
        Tests the case where there is an error in the vertical parameters.

        """

        with self.assertRaises(MissingParametersError):
            self._pvt(tilt=None)

    def test_horizontal_error(self) -> None:
        """
        Tests the case where there is an error in the horizontal parameters.

        """

        with self.assertRaises(MissingParametersError):
            self._pvt(azimuthal_orientation=None)

    def test_portion_covered_error(self) -> None:
        """
        Tests the case where there is an error in the portion-cover-related parameters.

        """

        with self.assertRaises(MissingParametersError):
            self._pvt(pv_params_provded=False)


class TestSolarAngle(_BaseTest):
    """
    Tests that the solar diference calculation is done correctly.

    The PVT system should compute, and return, the angle between the panel and the
    sun.

    """

    def test_dual_axis_tracking_solar_angle(self) -> None:
        """
        If the panel tracks on both axis, the angle should always be zero.

        """

        dual_axis_panel = self._pvt(horizontal_tracking=True, vertical_tracking=True)
        solar_angle_array = [(180, 50), (180, 90), (45, 50), (180, 90)]

        self.assertTrue(
            all(
                [
                    dual_axis_panel._get_solar_orientation_diff(  # pylint: disable=protected-access
                        *entry
                    )
                    == 0
                    for entry in solar_angle_array
                ]
            )
        )

    def test_horizontal_tracking_solar_angle(self) -> None:
        """
        Tests the angle computation if the panel tracks horizontally.

        """

        horizontally_tracking_panel = self._pvt(horizontal_tracking=True)

        # Assert that the panel returns the same angles regardless of azimuthal angle.
        self.assertEqual(
            horizontally_tracking_panel._get_solar_orientation_diff(  # pylint: disable=protected-access
                90, 45
            ),
            horizontally_tracking_panel._get_solar_orientation_diff(  # pylint: disable=protected-access
                180, 45
            ),
        )

    def test_vertical_tracking_solar_angle(self) -> None:
        """
        Tests the angle computation if the panel tracks vertically.

        """

        vertically_tracking_panel = self._pvt(vertical_tracking=True)

        # Assert that the panel returns the same angles regardless of the declination.
        self.assertEqual(
            vertically_tracking_panel._get_solar_orientation_diff(  # pylint: disable=protected-access
                50, 10
            ),
            vertically_tracking_panel._get_solar_orientation_diff(  # pylint: disable=protected-access
                50, 80
            ),
        )

    def test_angle_computation(self) -> None:
        """
        Tests that the angle computation takes place correctly.

        The angle computation is computed by:
            1. Determining the difference in the azimuthal angles and declinations of
            the sun and panel;
            2. Using cosines to project the solar vector onto the normal of the panel;
            3. Using an inverse cosine to determine the angle between the two.

        """

        pvt_panel = self._pvt(azimuthal_orientation=180, tilt=35)

        # Test when the sun is inline
        self.assertEqual(
            pvt_panel._get_solar_orientation_diff(  # pylint: disable=protected-access
                180, 35
            ),
            0,
        )

        # Fail if the PVT panel is incorrectly setup.
        if pvt_panel._azimuthal_orientation is None or pvt_panel._tilt is None:
            raise ProgrammerJudgementFault(
                "The azimuthal orientation and tilt on the PV panel were not set."
            )

        # Check a series of angles.
        solar_angle_array = [(180, 50), (180, 90), (45, 50), (180, 90)]
        self.assertEqual(
            [
                pvt_panel._get_solar_orientation_diff(  # pylint: disable=protected-access
                    azimuthal_angle, declination
                )
                for azimuthal_angle, declination in solar_angle_array
            ],
            [
                math.degrees(
                    math.acos(
                        math.cos(
                            abs(
                                math.radians(azimuthal_angle)
                                - math.radians(
                                    pvt_panel._azimuthal_orientation  # pylint: disable=protected-access
                                )
                            )
                        )
                        * math.cos(
                            abs(
                                math.radians(tilt)
                                - math.radians(
                                    pvt_panel._tilt  # pylint: disable=protected-access
                                )
                            )
                        )
                    )
                )
                for azimuthal_angle, tilt in solar_angle_array
            ],
        )


class TestIrradiance(_BaseTest):
    """
    Tests the internal solar irradiance computation function.

    """

    def test_get_solar_irradiance(self) -> None:
        """
        Tests that the calculation to determine the normal solar irradiance is correct.

        Depending on the method being used in the mainline code flow, the calculation
        of the solar irradiance may differ, in which case this test method will fail.

        """

        pvt_panel = self._pvt()
        weather_conditions = WeatherConditions(500, 0, 0, 0, 0, 0, 0)

        self.assertEqual(
            pvt_panel.get_solar_irradiance(weather_conditions),
            weather_conditions.irradiance,
        )


class TestProperties(_BaseTest):
    """
    Tests the various publicised internal properties of the :class:`pvt.PVT` class.

    """

    def test_coordinates(self) -> None:
        """
        Tests that the coordinates of the PVT panel are correctly publicised.

        """

        pvt_panel = self._pvt(latitude=45.5, longitude=10)

        self.assertEqual(pvt_panel.coordinates, (45.5, 10))

    def test_electrical_output(self) -> None:
        """
        Tests that the electrical output of the panel is correctly computed and returned

        """

        # Mock the PV layer's electrical efficiency.
        with mock.patch(
            "pvt_model.pvt_panel.pv.PV.electrical_efficiency", mock.MagicMock()
        ) as mock_pv_electrical_efficiency:
            mock_pv_electrical_efficiency.return_value = 0.5
            pvt_panel = self._pvt()
            weather_conditions = WeatherConditions(
                _irradiance=500,
                ambient_tank_temperature=0,
                ambient_temperature=0,
                azimuthal_angle=0,
                declination=25,
                mains_water_temperature=0,
                wind_speed=0,
            )
            self.assertEqual(
                pvt_panel.electrical_output(300, weather_conditions),
                0.5
                * weather_conditions.irradiance
                * pvt_panel.area
                * pvt_panel.portion_covered,
            )
