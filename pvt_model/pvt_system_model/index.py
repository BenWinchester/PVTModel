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


def _get_index(  # pylint: disable=too-many-return-statements,too-many-branches
    temperature_name: TemperatureName,
    *,
    number_of_pipes: Optional[int] = None,
    number_of_x_segments: Optional[int] = None,
    number_of_y_segments: Optional[int] = None,
    pipe_number: Optional[int] = None,
    x_coord: Optional[int] = None,
    y_coord: Optional[int] = None
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

    :param x_coord:
        The x coordinate of the segment whose index is being determined.

    :param y_coord:
        The y coordinate of the segment whose index is being determined.

    :return:
        A unique index describing the segment.

    :raises: ProgrammerJudegementFault
        This is raised if the method for defining the index for the temperature passed
        in is not defined in this method.

    """

    if temperature_name == TemperatureName.glass:
        if number_of_x_segments is None or x_coord is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "glass layer index."
            )
        return int(number_of_x_segments * y_coord + x_coord)
    if temperature_name == TemperatureName.pv:
        if (
            number_of_x_segments is None
            or number_of_y_segments is None
            or x_coord is None
            or y_coord is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "pv layer index."
            )
        return int(number_of_x_segments * (number_of_y_segments + y_coord) + x_coord)
    if temperature_name == TemperatureName.collector:
        if (
            number_of_x_segments is None
            or number_of_y_segments is None
            or x_coord is None
            or y_coord is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "absorber layer index."
            )
        return int(
            number_of_x_segments * (2 * number_of_y_segments + y_coord) + x_coord
        )
    if temperature_name == TemperatureName.pipe:
        if (
            number_of_x_segments is None
            or number_of_y_segments is None
            or pipe_number is None
            or y_coord is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "pipe index."
            )
        return int(
            (number_of_x_segments * (3 * number_of_y_segments + y_coord) + pipe_number)
        )
    if temperature_name == TemperatureName.htf:
        if (
            number_of_pipes is None
            or number_of_x_segments is None
            or number_of_y_segments is None
            or pipe_number is None
            or y_coord is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "htf index."
            )
        return int(
            (
                number_of_x_segments
                * (3 * number_of_y_segments + number_of_pipes + y_coord)
                + pipe_number
            )
        )
    if temperature_name == TemperatureName.htf_in:
        if (
            number_of_pipes is None
            or number_of_x_segments is None
            or number_of_y_segments is None
            or pipe_number is None
            or y_coord is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "htf-input index."
            )
        return int(
            (
                number_of_x_segments
                * (3 * number_of_y_segments + 2 * number_of_pipes + y_coord)
                + pipe_number
            )
        )
    if temperature_name == TemperatureName.htf_out:
        if (
            number_of_pipes is None
            or number_of_x_segments is None
            or number_of_y_segments is None
            or pipe_number is None
            or y_coord is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "htf-output index."
            )
        return int(
            (
                number_of_x_segments
                * (3 * number_of_y_segments + 3 * number_of_pipes + y_coord)
                + pipe_number
            )
        )
    if temperature_name == TemperatureName.collector_in:
        if (
            number_of_pipes is None
            or number_of_x_segments is None
            or number_of_y_segments is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "collector input index."
            )
        return int(
            number_of_x_segments * (3 * number_of_y_segments + 4 * number_of_pipes)
        )
    if temperature_name == TemperatureName.collector_out:
        if (
            number_of_pipes is None
            or number_of_x_segments is None
            or number_of_y_segments is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "collector output index."
            )
        return int(
            (
                number_of_x_segments * (3 * number_of_y_segments + 4 * number_of_pipes)
                + 1
            )
        )
    if temperature_name == TemperatureName.tank:
        if (
            number_of_pipes is None
            or number_of_x_segments is None
            or number_of_y_segments is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "tank index."
            )
        return int(
            (
                number_of_x_segments * (3 * number_of_y_segments + 4 * number_of_pipes)
                + 2
            )
        )
    if temperature_name == TemperatureName.tank_in:
        if (
            number_of_pipes is None
            or number_of_x_segments is None
            or number_of_y_segments is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "tank input index."
            )
        return int(
            (
                number_of_x_segments * (3 * number_of_y_segments + 4 * number_of_pipes)
                + 3
            )
        )
    if temperature_name == TemperatureName.tank_out:
        if (
            number_of_pipes is None
            or number_of_x_segments is None
            or number_of_y_segments is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "tank output index."
            )
        return int(
            (
                number_of_x_segments * (3 * number_of_y_segments + 4 * number_of_pipes)
                + 4
            )
        )
    raise ProgrammerJudgementFault(
        "An attempt was made to fetch an index using an undefined method."
    )


def index_from_pipe_coordinates(
    number_of_pipes: int,
    temperature_name: TemperatureName,
    pipe_number: int,
    y_coord: int,
) -> int:
    """
    Computes an index for a segmented pipe based on the coordinates of the segment.

    :param number_of_pipes:
        The number of HTF pipes in the collector.

    :param temperature_name:
        The name of the layer/temperature type being computed.

    :param pipe_number:
        The number of the pipe.

    :param y_coord:
        The y coordinate of the segment.

    :return:
        The index describing teh pipe and segment uniquely.

    """

    return _get_index(
        temperature_name,
        number_of_pipes=number_of_pipes,
        pipe_number=pipe_number,
        y_coord=y_coord,
    )


def index_from_segment_coordinates(
    number_of_x_segments: int,
    number_of_y_segments: int,
    temperature_name: TemperatureName,
    x_coord: int,
    y_coord: int,
) -> int:
    """
    Computes an index for a segmented layer based on the coordinates of the segment.

    :param number_of_x_segments:
        The number of segments in the x direction along the panel.

    :param number_of_y_segments:
        The number of segments in the y direction along the panel.

    :param temperature_name:
        The name of the layer/temperature type being computed.

    :param x_coord:
        The x coordinate of the segment.

    :param y_coord:
        The y coordinate of the segment.

    :return:
        The index describing the layer and the segment uniquely.

    """

    return _get_index(
        temperature_name,
        number_of_x_segments=number_of_x_segments,
        number_of_y_segments=number_of_y_segments,
        x_coord=x_coord,
        y_coord=y_coord,
    )


def index_from_temperature_name(temperature_name: TemperatureName) -> int:
    """
    Computes an index for a body/temperature based solely on the name.

    :param temperature_name:
        The name of the temperature/layer being computed.

    :return:
        An index uniquely describing the temperature name.

    """

    return _get_index(temperature_name)


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
