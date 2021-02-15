#!/usr/bin/python3.7
########################################################################################
# pvt_panel/test_collector.py - MUT for the Collector module.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Module Unit Tests (MUT) for the Collector module.

This module tests the various internal and external methods provided by the Collector
layer.

"""

import unittest

from unittest import mock  # pylint: disable=unused-import

import pytest
import numpy

from ...__utils__ import CollectorParameters
from .. import collector
from .test_utils import PYTEST_PRECISION


class TestProperties(unittest.TestCase):
    """
    Tests the various publically exposed private attributes and method properties.

    Some properties of the :class:`collector.Collector` instance are internally held
    and publically exposed through these methods, while other "properties" are
    determined by calculations within the class. This class contains test code that
    checks both kinds.

    """

    def setUp(self) -> None:
        """
        Sets up mocks in common across all test cases.

        """

        super().setUp()

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

        self.collector = collector.Collector(collector_parameters)

    def test_collector_to_htf_efficiency(self) -> None:
        """
        Tests the calculation for the efficiency for collector-to-htf efficiency.

        Tests that the correct calculation is done to determine the efficiency of the
        heat transfer process from the collector to the HTF in the riser tubes.

        """

    def test_convective_heat_transfer_coefficient_of_water(self) -> None:
        """
        Tests the heat transfer coefficient calculation.

        Tests that the correct calculation is done to determine the convective
        heat-transfer coefficient of water.

        """

    def test_htf_surface_area(self) -> None:
        """
        Tests that the correct calculation is done for the HTF surface area.

        The contact area, in meters squared, between the collector and the HTF should be
        computed. This is the internal area of the pipes in the collector.

        :equation:
            area = number_of_tubes * tube_length * PI * tube_diameter

        :units:
            m^2 = [dimensionless] * m * [dimensionless] * m

        """

        expected_htf_area = (
            self.collector.number_of_pipes
            * self.collector.length
            * numpy.pi
            * self.collector.pipe_diameter
        )

        self.assertEqual(
            pytest.approx(expected_htf_area, PYTEST_PRECISION),
            pytest.approx(self.collector.htf_surface_area, PYTEST_PRECISION),
        )

    def test_htf_volume(self) -> None:
        """
        Tests that the correct calculation is done for the HTF volume in the collector.

        The volume of HTF, in meters cubed, within the thermal collector riser tubes
        should be computed.

        :equation:
            volume = number_of_tubes * tube_length * PI * (tube_diameter / 2) ^ 2

        :units:
            m^3 = [dimensionless] * m * [dimensionless] * m ^2

        """

        expected_htf_volume = (
            self.collector.number_of_pipes
            * self.collector.length
            * numpy.pi
            * (self.collector.pipe_diameter / 2) ** 2
        )

        self.assertEqual(
            pytest.approx(expected_htf_volume, PYTEST_PRECISION),
            pytest.approx(self.collector.htf_volume, PYTEST_PRECISION),
        )

    def test_mass_flow_rate(self) -> None:
        """
        Tests that the correct internal calculation of the mass flow rate is done.

        """
