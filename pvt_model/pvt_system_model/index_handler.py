#!/usr/bin/python3.7
########################################################################################
# index_handler.py - The index solver module for the PVT model component.
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
    "temperature_name_from_index",
    "x_coordinate",
    "y_coordinate",
)


def _get_index(  # pylint: disable=too-many-branches
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
        The number of HTF pipes in the absorber.

    :param number_of_x_segments:
        The number of segments in the x direction for the absorber model being run.

    :param number_of_y_segments:
        The number of segments in the y direction for the absorber model being run.

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

    index: Optional[int] = None

    if temperature_name == TemperatureName.glass:
        if number_of_x_segments is None or x_coord is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "glass layer index_handler."
            )
        index = int(number_of_x_segments * y_coord + x_coord)
    if temperature_name == TemperatureName.pv:
        if (
            number_of_x_segments is None
            or number_of_y_segments is None
            or x_coord is None
            or y_coord is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "pv layer index_handler."
            )
        index = int(number_of_x_segments * (number_of_y_segments + y_coord) + x_coord)
    if temperature_name == TemperatureName.absorber:
        if (
            number_of_x_segments is None
            or number_of_y_segments is None
            or x_coord is None
            or y_coord is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "absorber layer index_handler."
            )
        index = int(
            number_of_x_segments * (2 * number_of_y_segments + y_coord) + x_coord
        )
    if temperature_name == TemperatureName.pipe:
        if (
            number_of_x_segments is None
            or number_of_y_segments is None
            or number_of_pipes is None
            or pipe_number is None
            or y_coord is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "pipe index_handler."
            )
        index = int(
            (
                3 * number_of_x_segments * number_of_y_segments
                + number_of_pipes * y_coord
                + pipe_number
            )
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
                "htf index_handler."
            )
        index = int(
            (
                (3 * number_of_x_segments + number_of_pipes) * number_of_y_segments
                + number_of_pipes * y_coord
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
                "htf-input index_handler."
            )
        index = int(
            (
                (3 * number_of_x_segments + 2 * number_of_pipes) * number_of_y_segments
                + number_of_pipes * y_coord
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
                "htf-output index_handler."
            )
        index = int(
            (
                (3 * number_of_x_segments + 3 * number_of_pipes) * number_of_y_segments
                + number_of_pipes * y_coord
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
                "absorber input index_handler."
            )
        index = int(
            (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments
        )
    if temperature_name == TemperatureName.collector_out:
        if (
            number_of_pipes is None
            or number_of_x_segments is None
            or number_of_y_segments is None
        ):
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine the "
                "absorber output index_handler."
            )
        index = int(
            (
                (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments
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
                "tank index_handler."
            )
        index = int(
            (
                (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments
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
                "tank input index_handler."
            )
        index = int(
            (
                (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments
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
                "tank output index_handler."
            )
        index = int(
            (
                (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments
                + 4
            )
        )

    # Return the index if assigned, else, raise an error.
    if index is not None:
        return index
    raise ProgrammerJudgementFault(
        "An attempt was made to fetch an index using an undefined method."
    )


def index_from_pipe_coordinates(
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    temperature_name: TemperatureName,
    pipe_number: int,
    y_coord: int,
) -> int:
    """
    Computes an index for a segmented pipe based on the coordinates of the segment.

    :param number_of_pipes:
        The number of HTF pipes in the absorber.

    :param number_of_x_segments:
        The number of segments in the x direction along the panel.

    :param number_of_y_segments:
        The number of segments in the y direction along the panel.

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
        number_of_x_segments=number_of_x_segments,
        number_of_y_segments=number_of_y_segments,
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


def index_from_temperature_name(
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    temperature_name: TemperatureName,
) -> int:
    """
    Computes an index for a body/temperature based solely on the name.

    :param number_of_pipes:
        The number of pipes in the absorber.

    :param number_of_x_segments:
        The number of segments in the x direction along the panel.

    :param number_of_y_segments:
        The number of segments in the y direction along the panel.

    :param temperature_name:
        The name of the temperature/layer being computed.

    :return:
        An index uniquely describing the temperature name.

    """

    return _get_index(
        number_of_pipes=number_of_pipes,
        number_of_x_segments=number_of_x_segments,
        number_of_y_segments=number_of_y_segments,
        temperature_name=temperature_name,
    )


def num_temperatures(
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
) -> int:
    """
    Returns the number of temperature variables being modelled.

    :param number_of_pipes:
        The number of pipes in the absorber.

    :param number_of_x_segments:
        The number of x segments in the absorber.

    :param number_of_y_segments:
        The number of y segments in the absorber.

    :return:
        The total number of temperatures being modelled.

    """

    return (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments + 5


def temperature_name_from_index(  # pylint: disable=too-many-branches
    index: int,
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
) -> TemperatureName:
    """
    Returns the temperature name from the index_handler.

    This method carries out a similar function in reverse to the `_get_index` method
    such that the temperature name, stored as a :class:`TemperatureName` instance, is
    returned based on the index passed in.

    :param index:
        The index of the temperature for which to return the temperature name.

    :param number_of_pipes:
        The number of HTF pipes in the absorber.

    :param number_of_x_segments:
        The number of segments in the x direction for the absorber model being run.

    :param number_of_y_segments:
        The number of segments in the y direction for the absorber model being run.

    :return:
        The temperature name, as a :class:`TemperatureName` instance, based on the index
        passed in.

    :raises: LookupError
        Raised when an index is passed in for which no temperature name has been
        assigned in this function.

    """

    temperature_name: Optional[TemperatureName] = None
    if index < number_of_x_segments * number_of_y_segments:
        temperature_name = TemperatureName.glass
    elif index < 2 * number_of_x_segments * number_of_y_segments:
        temperature_name = TemperatureName.pv
    elif index < 3 * number_of_x_segments * number_of_y_segments:
        temperature_name = TemperatureName.absorber
    elif index < ((3 * number_of_x_segments + number_of_pipes) * number_of_y_segments):
        temperature_name = TemperatureName.pipe
    elif index < (
        (3 * number_of_x_segments + 2 * number_of_pipes) * number_of_y_segments
    ):
        temperature_name = TemperatureName.htf
    elif index < (
        (3 * number_of_x_segments + 3 * number_of_pipes) * number_of_y_segments
    ):
        temperature_name = TemperatureName.htf_in
    elif index < (
        (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments
    ):
        temperature_name = TemperatureName.htf_out
    elif index == (
        (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments
    ):
        temperature_name = TemperatureName.collector_in
    elif index == (
        (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments + 1
    ):
        temperature_name = TemperatureName.collector_out
    elif index == (
        (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments + 2
    ):
        temperature_name = TemperatureName.tank
    elif index == (
        (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments + 3
    ):
        temperature_name = TemperatureName.tank_in
    elif index == (
        (3 * number_of_x_segments + 4 * number_of_pipes) * number_of_y_segments + 4
    ):
        temperature_name = TemperatureName.tank_out

    # Return the temperature name if assigned, else, return an error.
    if temperature_name is not None:
        return temperature_name
    raise LookupError(
        "A reverse lookup of the temperature name from the index was attempted: no "
        "matching temperature name was found."
    )


def x_coordinate(index: int, x_resolution: int) -> int:
    """
    Returns the x coordinate for the segment being processed from the index_handler.

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
    Returns the y coordinate for the segment being processed from the index_handler.

    :param index:
        The segment index being processed.

    :param x_resolution:
        The x resolution of the simulation being run.

    :return:
        The y corodinate of the segment.

    """

    return index // x_resolution
