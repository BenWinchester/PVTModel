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
