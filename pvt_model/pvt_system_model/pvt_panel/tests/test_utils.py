#!/usr/bin/python3.7
########################################################################################
# pvt_panel/test_utils.py - MUT for the utility module for the PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Module Unit Tests (MUT) for the PVT panel's utility module.

This module tests the various internal and external methods provided by the utility
module for the PVT panel.

"""

import unittest  # pylint: disable=unused-import

from unittest import mock  # pylint: disable=unused-import

import pytest  # pylint: disable=unused-import

from ...__utils__ import (  # pylint: disable=unused-import
    LayerParameters,
    OpticalLayerParameters,
)

from .. import __utils__

# The precision to be used in pytest tests.
PYTEST_PRECISION = 10 ** -6


class TestInstantiate(unittest.TestCase):
    """
    Tests the instantisation of the various dataclasses in the utility module.

    """

    def setUp(self) -> None:
        """
        Set up mocks and utilities in common across the test cases below.

        """

        super().setUp()
        self.layer_params = LayerParameters(
            mass=100,
            heat_capacity=200,
            area=15,
            thickness=0.001,
            density=2500,
        )
        self.optical_layer_params = OpticalLayerParameters(
            area=self.layer_params.area,
            conductivity=self.layer_params.conductivity,
            density=self.layer_params.density,
            heat_capacity=self.layer_params.heat_capacity,
            mass=self.layer_params.mass,
            thickness=self.layer_params.thickness,
            transmissivity=0.9,
            absorptivity=0.8,
            emissivity=0.7,
        )

    def test_instantiate_layer(self) -> None:
        """
        Tests the successful instantisation of a :class:`__utils__.Layer` instance.

        """

        layer = __utils__.Layer(self.layer_params)

        self.assertEqual(
            self.layer_params.mass, layer.mass  # pylint: disable=protected-access
        )
        self.assertEqual(
            self.layer_params.heat_capacity,
            layer.heat_capacity,  # pylint: disable=protected-access
        )
        self.assertEqual(self.layer_params.area, layer.area)
        self.assertEqual(self.layer_params.thickness, layer.thickness)

    def test_instantiate_optical_layer(self) -> None:
        """
        Tests the successful instantisation of a :class:`__utils__.OpticalLayer`.

        """

        optical_layer = __utils__.OpticalLayer(self.optical_layer_params)

        self.assertEqual(
            self.optical_layer_params.absorptivity, optical_layer.absorptivity
        )
        self.assertEqual(self.optical_layer_params.emissivity, optical_layer.emissivity)
        self.assertEqual(
            self.optical_layer_params.transmissivity, optical_layer.transmissivity
        )
