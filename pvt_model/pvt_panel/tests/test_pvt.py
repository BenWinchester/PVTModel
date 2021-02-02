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

import unittest

from unittest import mock

from .. import pvt


class TestInstantiate:
    """
    Tests the instantisation of the :class:`pvt.PVT` instance, probing all paths.

    """

    def test_mainline(self) -> None:
        """
        Tests the mainline case for PVT class instantisation.

        """

    def test_vertical_error(self) -> None:
        """
        Tests the case where there is an error in the vertical parameters.

        """

    def test_horizontal_error(self) -> None:
        """
        Tests the case where there is an error in the horizontal parameters.

        """

    def test_portion_covered_error(self) -> None:
        """
        Tests the case where there is an error in the portion-cover-related parameters.

        """


class TestInternals:
    """
    Tests the internal functions used by the :class:`pvt.PVT` class.

    """

    def test_solar_angle_diff(self) -> None:
        """
        Tests that the solar diference calculation is done correctly.

        """

    def test_get_solar_irradiance(self) -> None:
        """
        Tests that the calculation to determine the normal solar irradiance is correct.

        Depending on the method being used in the mainline code flow, the calculation
        of the solar irradiance may differ, in which case this test method will fail.

        """


class TestProperties:
    """
    Tests the various publicised internal properties of the :class:`pvt.PVT` class.

    """

    def test_bulk_water_temperature(self) -> None:
        """
        Tests that the bulk water temperature is correctly publicised.

        """

    def test_collector_output_temperature(self) -> None:
        """
        Tests that the collector output temperature is correctly publicised.

        """

    def test_collector_temperature(self) -> None:
        """
        Tests that the collector temperature is correctly publicised.

        """

    def test_coordinates(self) -> None:
        """
        Tests that the coordinates of the PVT panel are correctly publicised.

        """

    def test_electrical_efficiency(self) -> None:
        """
        Tests that the electrical efficiency of the PVT panel is correctly publicised.

        """

    def test_electrical_output(self) -> None:
        """
        Tests that the electrical output of the panel is correctly computed and returned

        """

    def test_glass_temperature(self) -> None:
        """
        Tests that the glass temperature is correctly publicised.

        """

    def test_glazed(self) -> None:
        """
        Tests whether the glazed property is computed correctly.

        """

    def test_htf_heat_capacity(self) -> None:
        """
        Tests that the heat capacity of the HTF in the collector layer is publicised.

        """

    def test_mass_flow_rate(self) -> None:
        """
        Tests that the mass flow rate of HTF through the collector is publicised.

        """

    def test_output_water_temperature(self) -> None:
        """
        Tests that the output water temperature of the collector is publicised.

        """

    def test_pump_power(self) -> None:
        """
        Tests that the power usage of the pump is correctly publicised.

        """

    def test_pv_temperature(self) -> None:
        """
        Tests that the temperature of the PV layer is correctly publicised.

        """


class TestUpdate:
    """
    Tests various flows through the :class:`pvt.PVT` instance's update method.

    """