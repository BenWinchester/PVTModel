#!/usr/bin/python3.7
########################################################################################
# pvt_panel/test_back_plate.py - MUT for the Back Plate module.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Module Unit Tests (MUT) for the Back Plate module.

This module tests the various internal and external methods provided by the back plate
layer.

"""

import unittest

from unittest import mock

import pytest

from ...__utils__ import BackLayerParameters
from ...constants import FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR
from .. import back_plate
from .test_utils import PYTEST_PRECISION


def test_instantiate() -> None:
    """
    Tests the instantisation and correct setting of various attributes of the back plate

    """


class TestProperties(unittest.TestCase):
    """
    Tests the various publically exposed private attributes and method properties.

    Some properties of the :class:`back_plate.BackPlate` instance are internally held
    and publically exposed through these methods, while other "properties" are
    determined by calculations within the class. This class contains test code that
    checks both kinds.

    """

    def test_conductance(self) -> None:
        """
        Tests that the calculation of the conductance is done correctly.

        :equation:
            conductnace = (
                thickness / conductivity
                + 1 / free_convective_heat_transfer_coefficient_of_air
            ) ^ (-1)

        :units:
            W/m^2*K = ((m / W/m*K) + (1 / W/m^2*K)) ^ (-1)

        :values:
            thickness = 0.01 m
            conductivity = 200 W/m*K
            free_convective_heat_transfer_coefficient_of_air = 25 W/m^2*K

        :result:
            conductace = 24.9687890137 W/m^2*K

        """

        # @@@ Switch to mocking once mocking is fixed.
        self.assertEqual(FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR, 25)

        test_back_plate = back_plate.BackPlate(
            BackLayerParameters(
                mass=100,
                heat_capacity=4000,
                area=15,
                thickness=0.01,
                conductivity=200,
            )
        )

        self.assertEqual(
            pytest.approx(test_back_plate.conductance, PYTEST_PRECISION),
            pytest.approx(24.9687890137, PYTEST_PRECISION),
        )
