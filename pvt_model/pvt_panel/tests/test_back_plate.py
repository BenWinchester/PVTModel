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

import unittest  # pylint: disable=unused-import

from unittest import mock  # pylint: disable=unused-import

from .. import back_plate  # pylint: disable=unused-import


def test_instantiate() -> None:
    """
    Tests the instantisation and correct setting of various attributes of the back plate

    """


class TestProperties:
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

        """
