#!/usr/bin/python3.7
########################################################################################
# pvt_collector/pipe.py - Represents a pipe within the system.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The pipe module for the PV-T model.

This module represents a pipe within the PV-T system.

"""

from dataclasses import dataclass

__all__ = ("Pipe",)


@dataclass
class Pipe:
    """
    Represents a pipe within the PVT system.

    .. attribute:: temperature
        The temperature of the pipe, measured in Kelvin.

    """

    temperature: float
