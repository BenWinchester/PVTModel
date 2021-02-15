#!/usr/bin/python3.7
########################################################################################
# pump.py - Represents a pump.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
Represents a pump.

"""

__all__ = ("Pump",)


class Pump:
    """
    Represents a physical pump in ths system.

    .. attribute:: on
        Whether the pump is currently on and pumping (True) or off (False). If the pump
        is off, there is no fluid flow through the pump.

    .. attribute:: power
        The electrical power consumed by the pump, measured in Watts.

    """

    def __init__(self, power: float) -> None:
        """
        Instantiates a :class:`Pump` instance.

        :param power:
            The power consumed by the pump when operating, measured in Watts.

        """

        self.on: bool = True
        self.power = power
