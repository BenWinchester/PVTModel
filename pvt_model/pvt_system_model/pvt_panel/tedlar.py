#!/usr/bin/python3.7
########################################################################################
# pvt_panel/tedlar.py - Represents an tedlar layer within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The tedlar module for the PV-T model.

This module represents a tedlar layer within a PV-T panel.

"""

from .__utils__ import OpticalLayer

__all__ = ("Tedlar",)


class Tedlar(OpticalLayer):
    """
    Represents an Tedlar layer within the panel.

    """
