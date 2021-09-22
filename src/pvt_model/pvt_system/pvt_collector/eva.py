#!/usr/bin/python3.7
########################################################################################
# pvt_collector/eva.py - Represents an eva layer within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The eva module for the PV-T model.

This module represents a eva layer within a PV-T panel.

"""

from .__utils__ import MicroLayer

__all__ = ("EVA",)


class EVA(MicroLayer):
    """
    Represents an EVA layer within the panel.

    """
