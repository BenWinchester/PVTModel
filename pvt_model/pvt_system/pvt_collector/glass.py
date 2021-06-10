#!/usr/bin/python3.7
########################################################################################
# pvt_collector/glass.py - Represents a glass within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The glass module for the PV-T model.

This module represents a glass layer within a PV-T panel.

"""

from .__utils__ import OpticalLayer

__all__ = ("Glass",)


class Glass(OpticalLayer):
    """
    Represents the glass (upper) layer of the PV-T panel.

    """
