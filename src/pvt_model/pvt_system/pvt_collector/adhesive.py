#!/usr/bin/python3.7
########################################################################################
# pvt_collector/adhesive_.py - Represents adhesive layer within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The adhesive_ module for the PV-T model.

This module represents a adhesive layer within a PV-T panel.

In addition to the upper glass cover represnted, there is scope for a small adhesive
layer placed in direct contact with the upper surface of the PV module.

"""

from .__utils__ import MicroLayer

__all__ = ("Adhesive",)


class Adhesive(MicroLayer):
    """
    Represents an Adhesive layer within the panel.

    """
