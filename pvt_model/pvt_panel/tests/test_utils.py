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

import unittest

from unittest import mock

from .. import __utils__


class TestInstantiate:
    """
    Tests the instantisation of the various dataclasses in the utility module.

    """

    def test_instantiate_layer(self) -> None:
        """
        Tests the successful instantisation of a :class:`__utils__.Layer` instance.

        """

    def test_instantiate_optical_layer(self) -> None:
        """
        Tests the successful instantisation of a :class:`__utils__.OpticalLayer`.

        """


class TestPhysics:
    """
    Tests the various Physics equations that are contained within the utility module.

    """

    def test_conductive_heat_transfer_no_gap(self) -> None:
        """
        Tests the publicised equation for the conductive heat transfer with no gap.

        """

    def test_conductive_heat_transfer_with_gap(self) -> None:
        """
        Tests the publicised equation for the conductive heat transfer with a gap.

        """

    def convective_heat_transfer_heat_transfer_to_fluid(self) -> None:
        """
        Tests the publicised equation for the convective heat transfer to a fluid.

        """

    def test_radiative_heat_transfer_between_bodies(self) -> None:
        """
        Tests the publicised equation for the radiative heat transfer between bodies.

        The method publicised can compute the radiative heat transfer between two
        parallel plates as well as that from a body to the sky/a sink at infinity.
        The radiative transfer between two plates is checked here.

        """

    def test_radiative_heat_transfer_to_sky(self) -> None:
        """
        Tests the publicised equation for the radiative heat transfer between bodies.

        The method publicised can compute the radiative heat transfer between two
        parallel plates as well as that from a body to the sky/a sink at infinity.
        The radiative transfer to the sky is checked here.

        """

    def test_solar_heat_input(self) -> None:
        """
        Tests the publicised equation for calculating the solar heat input.

        """

    def test_wind_heat_transfer(self) -> None:
        """
        Tests the publicised equation for the wind heat transfer.

        """