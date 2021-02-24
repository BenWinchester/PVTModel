#!/usr/bin/python3.7
########################################################################################
# pvt_panel/segment.py - Represents a single segment within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The segment module for the PV-T model.

This module represents a single segment within a PV-T panel.

"""

from dataclasses import dataclass

__all__ = ("Segment",)


@dataclass
class Segment:
    """
    Represents a single segment within a PV-T panel.

    .. attribute:: collector
        Whether the collector layer is present in this segment.

    .. attribute:: glass
        Whether the glass layer is present in this segment.

    .. attribute:: length
        The length of the segment, measured in meters.

    .. attribute:: pipe
        Whether there is a pipe attached to this layer.

    .. attribute:: pv
        Whether the pv layer is present in this segment.

    .. attribute:: width
        The width of the segment, measured in meters.

    .. attribute:: x_index
        The x index for this segment.

    .. attribute:: y_index
        The y index for this segment.

    """

    collector: bool
    glass: bool
    length: float
    pipe: bool
    pv: bool
    wdith: float
    x_index: float
    y_index: float
