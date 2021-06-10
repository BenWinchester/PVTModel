#!/usr/bin/python3.7
########################################################################################
# pvt_collector/bond.py - Represents an bond layer within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The bond module for the PV-T model.

This module represents a bond layer within a PV-T panel.

"""

from dataclasses import dataclass

from .__utils__ import MicroLayer

__all__ = ("Bond",)


@dataclass
class Bond(MicroLayer):
    """
    Represents an Bond layer within the panel.

    .. attribute:: width
        The width of the bond, measured in meters.

    """

    width: float
