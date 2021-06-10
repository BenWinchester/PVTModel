#!/usr/bin/python3.7
########################################################################################
# continuity.py - The continuity module for the matrix component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The continuity module for the matrix component.

This module computes and returns the equation(s) associated with the continuity
equations of the PV-T collector for the matrix component.

The model works by arranging the system of differential equations as a matrix equation
such that the temperatures at each time step can be computed based on the coefficients
of the matrix which are computed based on the temperatures of the various components at
the previous time step, as well as various Physical and fundamental constants.

"""


from typing import List, Tuple, Union

import numpy

from .. import index_handler
from ..pvt_collector import pvt

from ...__utils__ import (
    TemperatureName,
)
from ..pvt_collector.element import Element

__all__ = (
    "calculate_decoupled_system_continuity_equation",
    "calculate_fluid_continuity_equation",
    "calculate_system_continuity_equations",
)


def calculate_decoupled_system_continuity_equation(
    collector_input_temperature: float,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    number_of_y_elements: int,
    pvt_collector: pvt.PVT,
) -> List[Tuple[List[float], float]]:
    """
    Compute the system continuity equations when the system is decoupled.

    The equations represented and computed are:
        - fluid enters the absorber at the set temperature specified (1);
        - the fluid entering the absorber is the same across all pipes (2);
        - fluid leaving the absorber is computed as an average over all the outputs
          (2);

    """

    equations: List[Tuple[List[float], float]] = list()

    # Equation 1: Fluid enters the absorber at the temperature specified.
    row_equation: List[float] = [0] * number_of_temperatures
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.collector_in,
        )
    ] = 1
    equations.append((row_equation, collector_input_temperature))

    # Equation 2: Fluid entering the absorber is the same across all pipes.
    for pipe_number in range(number_of_pipes):
        row_equation = [0] * number_of_temperatures
        row_equation[
            index_handler.index_from_pipe_coordinates(
                number_of_pipes,
                number_of_x_elements,
                pipe_number,  # type: ignore
                pvt_collector,
                TemperatureName.htf_in,
                0,
            )
        ] = 1
        row_equation[
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.collector_in,
            )
        ] = -1
        equations.append((row_equation, 0))

    # Equation 3: Fluid leaving the absorber is computed by an average across the
    # output from all pipes.
    row_equation = [0] * number_of_temperatures
    for pipe_number in range(number_of_pipes):
        row_equation[
            index_handler.index_from_pipe_coordinates(
                number_of_pipes,
                number_of_x_elements,
                pipe_number,
                pvt_collector,
                TemperatureName.htf_out,
                number_of_y_elements - 1,
            )
        ] = (
            -1 / number_of_pipes
        )
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.collector_out,
        )
    ] = 1
    equations.append((row_equation, 0))

    return equations


def calculate_fluid_continuity_equation(
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    pvt_collector: pvt.PVT,
    element: Element,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing continuity of the htf.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    # Compute the T_f,out(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_elements,
            element.pipe_index,  # type: ignore
            pvt_collector,
            TemperatureName.htf_out,
            element.y_index,
        )
    ] = -1

    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_elements,
            element.pipe_index,  # type: ignore
            pvt_collector,
            TemperatureName.htf_in,
            element.y_index + 1,
        )
    ] = 1

    return row_equation, 0


def calculate_system_continuity_equations(
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    number_of_y_elements: int,
    previous_temperature_vector: Union[List[float], numpy.ndarray],
    pvt_collector: pvt.PVT,
) -> List[Tuple[List[float], float]]:
    """
    Returns matrix rows and resultant vector values representing system continuities.

    These inluce:
        - fluid entering the first section of the pipe is the same as that entering the
          absorber at the previous time step (1);
        - fluid leaving the absorber is an average over the various fluid temperatures
          leaving all pipes (2);
        - fluid entering the hot-water tank is the same as that leaving the absorber
          (3);
        - fluid leaving the hot-water tank is the same as that entering the absorber
          (4).

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `list` of `tuple`s containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    equations: List[Tuple[List[float], float]] = list()

    # Equation 1: Continuity of fluid entering the absorber.
    for pipe_number in range(number_of_pipes):
        row_equation: List[float] = [0] * number_of_temperatures
        row_equation[
            index_handler.index_from_pipe_coordinates(
                number_of_pipes,
                number_of_x_elements,
                pipe_number,
                pvt_collector,
                TemperatureName.htf_in,
                0,
            )
        ] = 1
        resultant_value = previous_temperature_vector[
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.collector_in,
            )
        ]
        equations.append((row_equation, resultant_value))

    # Equation 2: Fluid leaving the absorber is computed by an average across the
    # output from all pipes.
    row_equation = [0] * number_of_temperatures
    for pipe_number in range(number_of_pipes):
        row_equation[
            index_handler.index_from_pipe_coordinates(
                number_of_pipes,
                number_of_x_elements,
                pipe_number,
                pvt_collector,
                TemperatureName.htf_out,
                number_of_y_elements - 1,
            )
        ] = (
            -1 / number_of_pipes
        )
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.collector_out,
        )
    ] = 1
    equations.append((row_equation, 0))

    # Equation 3: Fluid leaving the absorber enters the tank without losses.
    row_equation = [0] * number_of_temperatures
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.collector_out,
        )
    ] = -1
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.tank_in,
        )
    ] = 1
    equations.append((row_equation, 0))

    row_equation = [0] * number_of_temperatures
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.tank_out,
        )
    ] = -1
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.collector_in,
        )
    ] = 1
    equations.append((row_equation, 0))

    return equations
