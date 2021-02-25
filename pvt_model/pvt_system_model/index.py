#!/usr/bin/python3.7
########################################################################################
# index.py - The index solver module for the PVT model component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The index module for the PVT model.

When solving the temperatures of the PVT model using a matrix method, it is necessary to
keep track of a mapping between the temperatures of the various components and a unique
index which can be used to represent the temperatures of the system as a single 1D
vector or `tuple`. The work for doing so is done here, in the index module.

"""

from typing import Optional

from ..__utils__ import (
    ProgrammerJudgementFault,
    TemperatureName,
)

__all__ = (
    "index_from_pipe_coordinates",
    "index_from_segment_coordinates",
    "index_from_temperature_name",
    "num_temperatures",
    "x_coordinate",
    "y_coordinate",
)


def _get_index(
    temperature_name: TemperatureName,
    *,
    number_of_pipes: Optional[int] = None,
    number_of_x_segments: Optional[int] = None,
    number_of_y_segments: Optional[int] = None,
    pipe_number: Optional[int] = None,
    x_coordinate: Optional[int] = None,
    y_coordinate: Optional[int] = None
) -> int:
    """
    Returns a unique index based on the inputs.

    :param temperature_name:
        The name of the temperature being indexed.

    :param number_of_pipes:
        The number of HTF pipes in the collector.

    :param number_of_x_segments:
        The number of segments in the x direction for the collector model being run.

    :param number_of_y_segments:
        The number of segments in the y direction for the collector model being run.

    :param pipe_number:
        The pipe number for which a coordinate should be returned.

    :param x_coordinate:
        The x coordinate of the segment whose index is being determined.

    :param y_coordinate:
        The y coordinate of the segment whose index is being determined.

    :return:
        A unique index describing the segment.

    :raises: ProgrammerJudegementFault
        This is raised if the method for defining the index for the temperature passed
        in is not defined in this method.

    """

    if temperature_name == TemperatureName.glass:
        try:
            return number_of_x_segments * y_coordinate + x_coordinate
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "glass layer index."
            ) from None
    if temperature_name == TemperatureName.pv:
        try:
            return (
                number_of_x_segments * (number_of_y_segments + y_coordinate)
                + x_coordinate
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "pv layer index."
            ) from None
    if temperature_name == TemperatureName.collector:
        try:
            return (
                number_of_x_segments * (2 * number_of_y_segments + y_coordinate)
                + x_coordinate
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "absorber layer index."
            ) from None
    if temperature_name == TemperatureName.pipe:
        try:
            return (
                number_of_x_segments * (3 * number_of_y_segments + y_coordinate)
                + pipe_number
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "pipe index."
            ) from None
    if temperature_name == TemperatureName.htf:
        try:
            return (
                number_of_x_segments
                * (3 * number_of_y_segments + number_of_pipes + y_coordinate)
                + pipe_number
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "htf index."
            ) from None
    if temperature_name == TemperatureName.htf_in:
        try:
            return (
                number_of_x_segments
                * (3 * number_of_y_segments + 2 * number_of_pipes + y_coordinate)
                + pipe_number
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "htf-input index."
            ) from None
    if temperature_name == TemperatureName.htf_out:
        try:
            return (
                number_of_x_segments
                * (3 * number_of_y_segments + 3 * number_of_pipes + y_coordinate)
                + pipe_number
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "htf-output index."
            ) from None
    if temperature_name == TemperatureName.collector_in:
        try:
            return number_of_x_segments * (
                3 * number_of_y_segments + 4 * number_of_pipes
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "collector input index."
            ) from None
    if temperature_name == TemperatureName.collector_out:
        try:
            return (
                number_of_x_segments * (3 * number_of_y_segments + 4 * number_of_pipes)
                + 1
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "collector output index."
            ) from None
    if temperature_name == TemperatureName.tank:
        try:
            return (
                number_of_x_segments * (3 * number_of_y_segments + 4 * number_of_pipes)
                + 2
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "tank index."
            ) from None
    if temperature_name == TemperatureName.tank_in:
        try:
            return (
                number_of_x_segments * (3 * number_of_y_segments + 4 * number_of_pipes)
                + 3
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "tank input index."
            ) from None
    if temperature_name == TemperatureName.tank_out:
        try:
            return (
                number_of_x_segments * (3 * number_of_y_segments + 4 * number_of_pipes)
                + 4
            )
        except TypeError:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "tank output index."
            ) from None
    raise ProgrammerJudgementFault(
        "An attempt was made to fetch an index using an undefined method."
    )


def index_from_pipe_coordinates(
    number_of_pipes: int, temperature_name: TemperatureName, pipe_number: int
) -> int:
    """
    Computes an index for a segmented pipe based on the coordinates of the segment.

    :param number_of_pipes:
        The number of HTF pipes in the collector.

    :param temperature_name:
        The name of the layer/temperature type being computed.

    :param pipe_number:
        The number of the pipe.

    :return:
        The index describing teh pipe and segment uniquely.

    """

    return _get_index(
        temperature_name, number_of_pipes=number_of_pipes, pipe_number=pipe_number
    )


def index_from_segment_coordinates(
    number_of_x_segments: int,
    number_of_y_segments: int,
    temperature_name: TemperatureName,
    x_coordinate: int,
    y_coordinate: int,
) -> int:
    """
    Computes an index for a segmented layer based on the coordinates of the segment.

    :param number_of_x_segments:
        The number of segments in the x direction along the panel.

    :param number_of_y_segments:
        The number of segments in the y direction along the panel.

    :param temperature_name:
        The name of the layer/temperature type being computed.

    :param x_coordinate:
        The x coordinate of the segment.

    :param y_coordinate:
        The y coordinate of the segment.

    :return:
        The index describing the layer and the segment uniquely.

    """

    _get_index(
        temperature_name,
        number_of_x_segments=number_of_x_segments,
        number_of_y_segments=number_of_y_segments,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )


def index_from_temperature_name(temperature_name: TemperatureName) -> int:
    """
    Computes an index for a body/temperature based solely on the name.

    :param temperature_name:
        The name of the temperature/layer being computed.

    :return:
        An index uniquely describing the temperature name.

    """

    _get_index(temperature_name)


def num_temperatures(
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
) -> int:
    """
    Returns the number of temperature variables being modelled.

    :param number_of_pipes:
        The number of pipes in the collector.

    :param number_of_x_segments:
        The number of x segments in the collector.

    :param number_of_y_segments:
        The number of y segments in the collector.

    :return:
        The total number of temperatures being modelled.

    """

    return (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments + 4


def x_coordinate(index: int, x_resolution: int) -> int:
    """
    Returns the x coordinate for the segment being processed from the index.

    :param index:
        The segment index being processed.

    :param x_resolution:
        The x resolution of the simulation being run.

    :return:
        The x corodinate of the segment.

    """

    return index % x_resolution


def y_coordinate(index: int, x_resolution: int) -> int:
    """
    Returns the y coordinate for the segment being processed from the index.

    :param index:
        The segment index being processed.

    :param x_resolution:
        The x resolution of the simulation being run.

    :return:
        The y corodinate of the segment.

    """

    return index // x_resolution
