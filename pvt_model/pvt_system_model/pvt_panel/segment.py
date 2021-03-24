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
from typing import Optional

__all__ = (
    "Segment",
    "SegmentCoordinates",
)


@dataclass
class SegmentCoordinates:
    """
    Represents the coordinates of the segment.

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

    .. attribute:: pipe_index
        The index of the attached pipe.

    """

    collector: bool
    glass: bool
    length: float
    pipe: bool
    pv: bool
    width: float
    x_index: int
    y_index: int
    pipe_index: Optional[int] = None

    def __str__(self) -> str:
        """
        Return a nice-looking representation of the segment.

        :return:
            A `str` giving a nice-looking representation of the segment.

        """

        layers = ", ".join(
            [
                entry
                for entry in [
                    "glass" if self.glass else None,
                    "collector" if self.collector else None,
                    "pv" if self.pv else None,
                    "pipe" if self.pipe else None,
                ]
                if entry is not None
            ]
        )

        return (
            "Segment("
            f"width: {self.width:.3f}m, "
            f"length: {self.length:.3f}m, "
            f"layers: {layers}"
            ")"
        )

    @property
    def coordinates(self) -> SegmentCoordinates:
        """
        Returns the coordinates of the segment as a segment coordinates object.

        :return:
            The segment coordinates.

        """

        return SegmentCoordinates(self.x_index, self.y_index)
