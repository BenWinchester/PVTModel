#!/usr/bin/python3.7
########################################################################################
# pvt_panel/eva.py - Represents an eva layer within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The eva module for the PV-T model.

This module represents a eva layer within a PV-T panel.

"""

from .__utils__ import OpticalLayer

__all__ = ("EVA",)


class EVA(OpticalLayer):
    """
    Represents an EVA layer within the panel.

    """
