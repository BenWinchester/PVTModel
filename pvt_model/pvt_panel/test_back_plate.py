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

from unittest import mock

from . import collector


def test_instantiate() -> None:
    """
    Tests the instantisation of a :class:`collector.Collector` instance.

    This checks that all private attributes are set as expected.

    """


class TestProperties:
    """
    Tests the various publically exposed private attributes and method properties.

    Some properties of the :class:`collector.Collector` instance are internally held
    and publically exposed through these methods, while other "properties" are
    determined by calculations within the class. This class contains test code that
    checks both kinds.

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

        """

    def test_htf_volume(self) -> None:
        """
        Tests that the correct calculation is done for the HTF volume in the collector.

        """

    def test_mass_flow_rate(self) -> None:
        """
        Tests that the correct internal calculation of the mass flow rate is done.

        """


class TestUpdate:
    """
    Tests the internal update method of :class:`collector.Collector` instances.

    The update method has several different flows depending on the upper layers. This
    class contains methods that probe all of these paths, as well as exceptions that
    may occur.

    """

    def test_update_no_pv_no_glass(self) -> None:
        """
        Tests the update case where there are neither PV nor glass layers above.

        """

    def test_update_some_pv_no_glass(self) -> None:
        """
        Tests the update case where there is a partial PV layer but no glass layer.

        """

    def test_update_all_pv_no_glass(self) -> None:
        """
        Tests the udpate case where the collector is fully covered by a PV layer.

        """

    def test_update_no_pv_with_glass(self) -> None:
        """
        Tests the collector's update method when covered with a glass layer only.

        Tests the update case where there is no PV layer but the collector is fully
        covered by a glass layer.

        """

    def test_update_some_pv_with_glass(self) -> None:
        """
        Tests the collector's update method when covered with part-PV and part-glass.

        Tests the update case where there is a partial PV layer and the remainder of the
        collector laer is covered by a glass layer.

        """

    def test_update_all_pv_with_glass(self) -> None:
        """
        Tests the collector's update method when fully-covered with PV.

        Tests the update case where the collector is fully covered by a PV layer but
        where a glass layer is still present.

        NOTE: In this instance, the collector code should not supply any upward heat to
        the glass layer.

        """

    def test_melt(self) -> None:
        """
        Tests that, when the collector melts, the pdb debugger is called.

        """
