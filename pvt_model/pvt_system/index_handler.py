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

from typing import Optional, Tuple

from ..__utils__ import (
    ProgrammerJudgementFault,
    TemperatureName,
)
from .pvt_collector import pvt

__all__ = (
    "index_from_pipe_coordinates",
    "index_from_element_coordinates",
    "index_from_temperature_name",
    "num_temperatures",
    "temperature_name_from_index",
    "x_coordinate",
    "y_coordinate",
)


def _calculate_number_of_temperatures(
    pvt_collector: pvt.PVT,
) -> Tuple[int, int, int, int, int, int]:
    """
    Calculates, based off the PVT collector, the number of temperatures in each layer.

    :param pvt_collector:
        The PVT collector being modelled.

    :return:
        A `tuple` containing:
        - the number of upper-glass (i.e., double-glazing) temperatures in the
          collector,
        - the number of glass temperatures in the collector,
        - the number of pv temperatures in the collector,
        - the number of absorber temperatures in the collector,
        - the number of panel (i.e., glass, pv, and absorber) temperatures in the
          collector,
        - the number of pipe/htf temperatures in the collector,

    """

    num_upper_glass_temperatures = sum(
        [element.upper_glass for element in pvt_collector.elements.values()]
    )
    num_glass_temperatures = sum(
        [element.glass for element in pvt_collector.elements.values()]
    )
    num_pv_temperatures = sum(
        [element.pv for element in pvt_collector.elements.values()]
    )
    num_absorber_temperatures = sum(
        [element.absorber for element in pvt_collector.elements.values()]
    )
    num_panel_temperatures: int = (
        num_upper_glass_temperatures
        + num_glass_temperatures
        + num_pv_temperatures
        + num_absorber_temperatures
    )
    num_fluid_temperatures = sum(
        [element.pipe for element in pvt_collector.elements.values()]
    )

    return (
        num_upper_glass_temperatures,
        num_glass_temperatures,
        num_pv_temperatures,
        num_absorber_temperatures,
        num_panel_temperatures,
        num_fluid_temperatures,
    )


def _get_index(  # pylint: disable=too-many-branches
    temperature_name: TemperatureName,
    pvt_collector: pvt.PVT,
    *,
    number_of_pipes: Optional[int] = None,
    number_of_x_elements: Optional[int] = None,
    pipe_number: Optional[int] = None,
    x_coord: Optional[int] = None,
    y_coord: Optional[int] = None
) -> int:
    """
    Returns a unique index based on the inputs.

    :param temperature_name:
        The name of the temperature being indexed.

    :param pvt_collector:
        The :class:`pvt.PVT` instance representing the PVT collector being modelled.

    :param number_of_pipes:
        The number of HTF pipes in the absorber.

    :param number_of_x_elements:
        The number of elements in the x direction for the absorber model being run.

    :param pipe_number:
        The pipe number for which a coordinate should be returned.

    :param x_coord:
        The x coordinate of the element whose index is being determined.

    :param y_coord:
        The y coordinate of the element whose index is being determined.

    :return:
        A unique index describing the element.

    :raises: ProgrammerJudegementFault
        This is raised if the method for defining the index for the temperature passed
        in is not defined in this method.

    """

    index: Optional[int] = None

    # Fetch the number of temperatures of each type.
    (
        num_upper_glass_temperatures,
        num_glass_temperatures,
        num_pv_temperatures,
        _,
        num_panel_temperatures,
        num_fluid_temperatures,
    ) = _calculate_number_of_temperatures(pvt_collector)

    if temperature_name == TemperatureName.upper_glass:
        if number_of_x_elements is None or x_coord is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "upper-glass (i.e., double-glazing) layer index_handler."
            )
        index = int(number_of_x_elements * y_coord + x_coord)
    if temperature_name == TemperatureName.glass:
        if number_of_x_elements is None or x_coord is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "glass layer index_handler."
            )
        index = num_upper_glass_temperatures + int(
            number_of_x_elements * y_coord + x_coord
        )
    if temperature_name == TemperatureName.pv:
        if number_of_x_elements is None or x_coord is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "pv layer index_handler."
            )
        index = (
            num_upper_glass_temperatures
            + num_glass_temperatures
            + int(number_of_x_elements * y_coord + x_coord)
        )
    if temperature_name == TemperatureName.absorber:
        if number_of_x_elements is None or x_coord is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "absorber layer index_handler."
            )
        index = (
            num_upper_glass_temperatures
            + num_glass_temperatures
            + num_pv_temperatures
            + int(number_of_x_elements * y_coord + x_coord)
        )
    if temperature_name == TemperatureName.pipe:
        if number_of_pipes is None or pipe_number is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine a "
                "pipe index_handler."
            )
        index = num_panel_temperatures + int((number_of_pipes * y_coord + pipe_number))
    if temperature_name == TemperatureName.htf:
        if number_of_pipes is None or pipe_number is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "htf index_handler."
            )
        index = (
            num_panel_temperatures
            + num_fluid_temperatures
            + int((number_of_pipes * y_coord + pipe_number))
        )
    if temperature_name == TemperatureName.htf_in:
        if number_of_pipes is None or pipe_number is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "htf-input index_handler."
            )
        index = (
            num_panel_temperatures
            + 2 * num_fluid_temperatures
            + int((number_of_pipes * y_coord + pipe_number))
        )
    if temperature_name == TemperatureName.htf_out:
        if number_of_pipes is None or pipe_number is None or y_coord is None:
            raise ProgrammerJudgementFault(
                "Not all parameters needed were passed in to uniquely determine an "
                "htf-output index_handler."
            )
        index = (
            num_panel_temperatures
            + 3 * num_fluid_temperatures
            + int((number_of_pipes * y_coord + pipe_number))
        )
    if temperature_name == TemperatureName.collector_in:
        index = num_panel_temperatures + 4 * num_fluid_temperatures
    if temperature_name == TemperatureName.collector_out:
        index = num_panel_temperatures + 4 * num_fluid_temperatures + 1
    if temperature_name == TemperatureName.tank:
        index = num_panel_temperatures + 4 * num_fluid_temperatures + 2
    if temperature_name == TemperatureName.tank_in:
        index = num_panel_temperatures + 4 * num_fluid_temperatures + 3
    if temperature_name == TemperatureName.tank_out:
        index = num_panel_temperatures + 4 * num_fluid_temperatures + 4

    # Return the index if assigned, else, raise an error.
    if index is not None:
        return index
    raise ProgrammerJudgementFault(
        "An attempt was made to fetch an index using an undefined method."
    )


def index_from_pipe_coordinates(
    number_of_pipes: int,
    number_of_x_elements: int,
    pipe_number: int,
    pvt_collector: pvt.PVT,
    temperature_name: TemperatureName,
    y_coord: int,
) -> int:
    """
    Computes an index for a elemented pipe based on the coordinates of the element.

    :param number_of_pipes:
        The number of HTF pipes in the absorber.

    :param number_of_x_elements:
        The number of elements in the x direction along the panel.

    :param pipe_number:
        The number of the pipe.

    :param pvt_collector:
        The pvt collector being modelled.

    :param temperature_name:
        The name of the layer/temperature type being computed.

    :param y_coord:
        The y coordinate of the element.

    :return:
        The index describing teh pipe and element uniquely.

    """

    return _get_index(
        temperature_name,
        pvt_collector,
        number_of_pipes=number_of_pipes,
        number_of_x_elements=number_of_x_elements,
        pipe_number=pipe_number,
        y_coord=y_coord,
    )


def index_from_element_coordinates(
    number_of_x_elements: int,
    pvt_collector: pvt.PVT,
    temperature_name: TemperatureName,
    x_coord: int,
    y_coord: int,
) -> int:
    """
    Computes an index for a elemented layer based on the coordinates of the element.

    :param number_of_x_elements:
        The number of elements in the x direction along the panel.

    :param pvt_collector:
        The pvt collector being modelled.

    :param temperature_name:
        The name of the layer/temperature type being computed.

    :param x_coord:
        The x coordinate of the element.

    :param y_coord:
        The y coordinate of the element.

    :return:
        The index describing the layer and the element uniquely.

    """

    return _get_index(
        temperature_name,
        pvt_collector,
        number_of_x_elements=number_of_x_elements,
        x_coord=x_coord,
        y_coord=y_coord,
    )


def index_from_temperature_name(
    pvt_collector: pvt.PVT,
    temperature_name: TemperatureName,
) -> int:
    """
    Computes an index for a body/temperature based solely on the name.

    :param pvt_collector:
        The PVT collector being modelled.

    :param temperature_name:
        The name of the temperature/layer being computed.

    :return:
        An index uniquely describing the temperature name.

    """

    return _get_index(
        temperature_name,
        pvt_collector,
    )


def num_temperatures(pvt_collector: pvt.PVT) -> int:
    """
    Returns the number of temperature variables being modelled.

    :param pvt_collector:
        The pvt panel being modelled.

    :return:
        The total number of temperatures being modelled.

    """

    return (
        sum(
            [
                element.upper_glass
                + element.glass
                + element.pv
                + element.absorber
                + 4 * element.pipe
                for element in pvt_collector.elements.values()
            ]
        )
        + 5
    )


def temperature_name_from_index(  # pylint: disable=too-many-branches
    index: int,
    pvt_collector: pvt.PVT,
) -> TemperatureName:
    """
    Returns the temperature name from the index_handler.

    This method carries out a similar function in reverse to the `_get_index` method
    such that the temperature name, stored as a :class:`TemperatureName` instance, is
    returned based on the index passed in.

    :param index:
        The index of the temperature for which to return the temperature name.

    :param pvt_collector:
        The pvt collector being modelled.

    :return:
        The temperature name, as a :class:`TemperatureName` instance, based on the index
        passed in.

    :raises: LookupError
        Raised when an index is passed in for which no temperature name has been
        assigned in this function.

    """

    temperature_name: Optional[TemperatureName] = None

    # Determine the number of temperatures of each type.
    (
        num_upper_glass_temperatures,
        num_glass_temperatures,
        num_pv_temperatures,
        _,
        num_panel_temperatures,
        num_fluid_temperatures,
    ) = _calculate_number_of_temperatures(pvt_collector)

    if index < num_upper_glass_temperatures:
        temperature_name = TemperatureName.upper_glass
    elif index < num_upper_glass_temperatures + num_glass_temperatures:
        temperature_name = TemperatureName.glass
    elif (
        index
        < num_upper_glass_temperatures + num_glass_temperatures + num_pv_temperatures
    ):
        temperature_name = TemperatureName.pv
    elif index < num_panel_temperatures:
        temperature_name = TemperatureName.absorber
    elif index < num_panel_temperatures + num_fluid_temperatures:
        temperature_name = TemperatureName.pipe
    elif index < (num_panel_temperatures + 2 * num_fluid_temperatures):
        temperature_name = TemperatureName.htf
    elif index < (num_panel_temperatures + 3 * num_fluid_temperatures):
        temperature_name = TemperatureName.htf_in
    elif index < (num_panel_temperatures + 4 * num_fluid_temperatures):
        temperature_name = TemperatureName.htf_out
    elif index == (num_panel_temperatures + 4 * num_fluid_temperatures):
        temperature_name = TemperatureName.collector_in
    elif index == (num_panel_temperatures + 4 * num_fluid_temperatures + 1):
        temperature_name = TemperatureName.collector_out
    elif index == (num_panel_temperatures + 4 * num_fluid_temperatures + 2):
        temperature_name = TemperatureName.tank
    elif index == (num_panel_temperatures + 4 * num_fluid_temperatures + 3):
        temperature_name = TemperatureName.tank_in
    elif index == (num_panel_temperatures + 4 * num_fluid_temperatures + 4):
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
    Returns the x coordinate for the element being processed from the index_handler.

    :param index:
        The element index being processed.

    :param x_resolution:
        The x resolution of the simulation being run.

    :return:
        The x corodinate of the element.

    """

    return index % x_resolution


def y_coordinate(index: int, x_resolution: int) -> int:
    """
    Returns the y coordinate for the element being processed from the index_handler.

    :param index:
        The element index being processed.

    :param x_resolution:
        The x resolution of the simulation being run.

    :return:
        The y corodinate of the element.

    """

    return index // x_resolution
