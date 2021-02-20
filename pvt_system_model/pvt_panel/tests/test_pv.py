#!/usr/bin/python3.7
########################################################################################
# pvt_panel/test_pv.py - MUT for the PV module.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Module Unit Tests (MUT) for the PV module.

This module tests the various internal and external methods provided by the PV panel.

"""

import unittest

import pytest

from ...__utils__ import PVParameters
from .. import pv
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

        self.pv_parameters = PVParameters(
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

        self.pv = pv.PV(self.pv_parameters)


class TestInstantiate(_BaseTest):
    """
    Tests the instantiation of the :class:`pv.PV` instance.

    """

    def test_instantiate(self) -> None:
        """
        Tests the instantisation of a :class:`pv.PV` instance.

        This checks that all private attributes are set as expected.

        """

        self.assertEqual(self.pv.mass, self.pv_parameters.mass)
        self.assertEqual(self.pv.heat_capacity, self.pv_parameters.heat_capacity)
        self.assertEqual(self.pv.area, self.pv_parameters.area)
        self.assertEqual(self.pv.thickness, self.pv_parameters.thickness)
        self.assertEqual(self.pv.transmissivity, self.pv_parameters.transmissivity)
        self.assertEqual(self.pv.absorptivity, self.pv_parameters.absorptivity)
        self.assertEqual(self.pv.emissivity, self.pv_parameters.emissivity)
        self.assertEqual(
            self.pv.reference_efficiency, self.pv_parameters.reference_efficiency
        )
        self.assertEqual(
            self.pv.reference_temperature, self.pv_parameters.reference_temperature
        )
        self.assertEqual(
            self.pv.thermal_coefficient, self.pv_parameters.thermal_coefficient
        )


class TestProperties(_BaseTest):
    """
    Tests various properties of the :class:`pv.PV` instance.

    """

    def test_electrical_efficiency(self) -> None:
        """
        Tests the electrical efficiency calculation within of a :class:`pv.PV` instances.

        Photovoltaic cells produce an electrical output. An inherent property of a PV cell,
        or layer, is its electrical efficiency. This depends on the temperature, and so is
        an internal function within the PV layer and is dependant on its temperature.

        :equation:
            electrical_efficiency = reference_electrical_efficiency * [
                1 - thermal_coefficient * (
                    temperature_of_the_pv_layer - reference_pv_temperature
                )
            ]

        :units:
            [unitless] = [unitless] * (1 - K^-1) * K

        :values:
            temperature_of_the_pv_layer = 350 K

        """

        expected_electrical_efficiency = self.pv.reference_efficiency * (
            1 - self.pv.thermal_coefficient * (350 - self.pv.reference_temperature)
        )

        self.assertEqual(
            pytest.approx(self.pv.electrical_efficiency(350), PYTEST_PRECISION),
            pytest.approx(expected_electrical_efficiency, PYTEST_PRECISION),
        )
