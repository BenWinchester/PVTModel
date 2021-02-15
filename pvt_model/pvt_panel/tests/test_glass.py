#!/usr/bin/python3.7
########################################################################################
# pvt_panel/test_glass.py - MUT for the Glass module.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Module Unit Tests (MUT) for the Glass module.

This module tests the various internal and external methods provided by the glass layer
of the panel.

"""

import unittest

from unittest import mock  # pylint: disable=unused-import

import pytest

from ...__utils__ import OpticalLayerParameters
from .. import glass
from .test_utils import PYTEST_PRECISION


class _BaseTest(unittest.TestCase):
    """
    Contains base functionality to set up mocks and functions in common across tests.

    """

    def setUp(self) -> None:
        """
        Sets up mocks in common across test cases.

        """

        super().setUp()

        self.glass_parameterse = OpticalLayerParameters(
            mass=150,
            heat_capacity=4000,
            area=100,
            thickness=0.015,
            transmissivity=0.9,
            absorptivity=0.88,
            emissivity=0.5,
        )
        self.diffuse_reflection_coefficient = 0.18

        self.glass = glass.Glass(
            self.diffuse_reflection_coefficient, self.glass_parameterse
        )


class TestInstantiate(_BaseTest):
    """
    Tests the instantiation of the :class:`pv.PV` instance.

    """

    def test_instantiate(self) -> None:
        """
        Tests the instantisation of a :class:`pv.PV` instance.

        This checks that all private attributes are set as expected.

        """

        self.assertEqual(
            self.glass.diffuse_reflection_coefficient,
            self.diffuse_reflection_coefficient,
        )
        self.assertEqual(self.glass.mass, self.glass_parameterse.mass)
        self.assertEqual(self.glass.heat_capacity, self.glass_parameterse.heat_capacity)
        self.assertEqual(self.glass.area, self.glass_parameterse.area)
        self.assertEqual(self.glass.thickness, self.glass_parameterse.thickness)
        self.assertEqual(
            self.glass.transmissivity, self.glass_parameterse.transmissivity
        )
        self.assertEqual(self.glass.absorptivity, self.glass_parameterse.absorptivity)
        self.assertEqual(self.glass.emissivity, self.glass_parameterse.emissivity)
