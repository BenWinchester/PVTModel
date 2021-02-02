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

from unittest import mock

from . import pv


def test_instantiate() -> None:
    """
    Tests the instantisation of a :class:`pv.PV` instance.

    This checks that all private attributes are set as expected.

    """


def test_electrical_efficiency() -> None:
    """
    Tests the electrical efficiency calculation within of a :class:`pv.PV` instances.

    Photovoltaic cells produce an electrical output. An inherent property of a PV cell,
    or layer, is its electrical efficiency. This depends on the temperature, and so is
    an internal function within the PV layer and is dependant on its temperature.

    The electrical efficiency is hence computed for a dummy PV layer, using the
    following formula and values:

    electrical_efficiency = reference_electrical_efficiency * [
        1 - thermal_coefficient * (
            temperature_of_the_pv_layer - reference_pv_temperature
        )
    ]

    """


class TestUpdate:
    """
    Tests the internal update method of :class:`pv.PV` instances.

    The update method has several different flows depending on the other layers. This
    class contains methods that probe all of these paths, as well as exceptions that
    may occur.

    """

    def test_update_with_glass_layer(self) -> None:
        """
        Tests the update method with a glass layer being present.

        Depending on whether a glass layer is present, the updating behaviour of the PV
        layer will be different, and the calls to various utility modules will be different.

        """

    def test_update_no_glass_layer(self) -> None:
        """
        Tests the update method with a glass layer being present.

        Depending on whether a glass layer is present, the updating behaviour of the PV
        layer will be different, and the calls to various utility modules will be different.

        """
