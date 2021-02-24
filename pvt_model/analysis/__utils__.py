#!/usr/bin/python3.7
########################################################################################
# __utils__.py - The utility module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The utility module for the analysis component.

"""

import enum

__all__ = ("GraphDetail",)


class GraphDetail(enum.Enum):
    """
    The level of detail to go into when graphing.

    .. attribute:: highest
        The highest level of detail - all data points are plotted.

    .. attribute:: high
        A "high" level of detail, to be determined by the analysis script.

    .. attribute:; medium
        A "medium" level of detail, to be determined by the analysis script.

    .. attribute:: low
        A "low" level of detail, to be determined by the analysis script.

    .. attribute:: lowest
        The lowest level of detail, with points only every half an hour.

    """

    highest = 0
    high = 2880
    medium = 720
    low = 144
    lowest = 48
