#!/usr/bin/python3.7
########################################################################################
# pvt_collector/element.py - Represents a single element within a PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The element module for the PV-T model.

This module represents a single element within a PV-T panel.

"""

from dataclasses import dataclass
from typing import Optional

__all__ = (
    "Element",
    "ElementCoordinates",
)


@dataclass
class ElementCoordinates:
    """
    Represents the coordinates of the element.

    .. attribute:: x_index
        The x index of the coordinate.

    .. attribute:: y_index
        The y_index of the coordinate.

    """

    x_index: int
    y_index: int

    def __hash__(self) -> int:
        """
        Returns a unique hash based on the two coordinates.

        :return:
            A unique number representing the two coordiantes.

        """

        return int(
            ((self.x_index + self.y_index) * (self.x_index + self.y_index + 1) / 2)
            + self.x_index
        )

    def __str__(self) -> str:
        """
        Return a string representing the coordinates.

        :return:
            A `str` detailing the coordinate information.

        """

        return f"({self.x_index}, {self.y_index})"


@dataclass
class Element:
    """
    Represents a single element within a PV-T panel.

    .. attribute:: absorber
        Whether the absorber layer is present in this element.

    .. attribute:: glass
        Whether the glass layer is present in this element.

    .. attribute:: length
        The length of the element, measured in meters.

    .. attribute:: pipe
        Whether there is a pipe attached to this layer.

    .. attribute:: pv
        Whether the pv layer is present in this element.

    .. attribute:: upper_glass
        Whether an upper-glass layer (i.e., double-glazing) is present.

    .. attribute:: width
        The width of the element, measured in meters.

    .. attribute:: x_index
        The x index for this element.

    .. attribute:: y_index
        The y index for this element.

    .. attribute:: pipe_index
        The index of the attached pipe.

    """

    absorber: bool
    glass: bool
    length: float
    pipe: bool
    pv: bool
    upper_glass: bool
    width: float
    x_index: int
    y_index: int
    pipe_index: Optional[int] = None

    def __str__(self) -> str:
        """
        Return a nice-looking representation of the element.

        :return:
            A `str` giving a nice-looking representation of the element.

        """

        layers = ", ".join(
            [
                entry
                for entry in [
                    "upper-glass" if self.upper_glass else None,
                    "glass" if self.glass else None,
                    "absorber" if self.absorber else None,
                    "pv" if self.pv else None,
                    "pipe" if self.pipe else None,
                ]
                if entry is not None
            ]
        )

        return (
            "Element("
            f"width: {self.width:.3f}m, "
            f"length: {self.length:.3f}m, "
            f"layers: {layers}"
            ")"
        )

    @property
    def coordinates(self) -> ElementCoordinates:
        """
        Returns the coordinates of the element as a element coordinates object.

        :return:
            The element coordinates.

        """

        return ElementCoordinates(self.x_index, self.y_index)
