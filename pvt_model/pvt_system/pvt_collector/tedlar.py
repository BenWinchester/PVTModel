#!/usr/bin/python3.7
########################################################################################
# pvt_collector/tedlar.py - Represents an tedlar layer within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The tedlar module for the PV-T model.

This module represents a tedlar layer within a PV-T panel.

"""

from .__utils__ import MicroLayer

__all__ = ("Tedlar",)


class Tedlar(MicroLayer):
    """
    Represents an Tedlar layer within the panel.

    """
