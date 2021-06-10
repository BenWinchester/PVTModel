#!/usr/bin/python3.7
########################################################################################
# htf.py - The htf module for the matrix component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The htf module for the matrix component.

This module computes and returns the equation(s) associated with the htf layer of the
PV-T collector for the matrix component.

The model works by arranging the system of differential equations as a matrix equation
such that the temperatures at each time step can be computed based on the coefficients
of the matrix which are computed based on the temperatures of the various components at
the previous time step, as well as various Physical and fundamental constants.

"""


from typing import List, Optional, Tuple, Union

import numpy

from .. import index_handler
from ..pvt_collector import pvt

from ...__utils__ import (
    OperatingMode,
    TemperatureName,
)
from ..physics_utils import density_of_water
from ..pvt_collector.element import Element

__all__ = ("calculate_htf_continuity_equation", "calculate_htf_equation")


def calculate_htf_continuity_equation(
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    pvt_collector: pvt.PVT,
    element: Element,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing the htf equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    # Compute the T_f(#, 0) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_elements,
            element.pipe_index,  # type: ignore
            pvt_collector,
            TemperatureName.htf,
            element.y_index,
        )
    ] = 1

    # Compute the T_f,in(#, 0) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_elements,
            element.pipe_index,  # type: ignore
            pvt_collector,
            TemperatureName.htf_in,
            element.y_index,
        )
    ] = -0.5

    # Compute the T_f,out(#, 0) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_elements,
            element.pipe_index,  # type: ignore
            pvt_collector,
            TemperatureName.htf_out,
            element.y_index,
        )
    ] = -0.5

    return row_equation, 0


def calculate_htf_equation(
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    operating_mode: OperatingMode,
    pipe_to_bulk_water_heat_transfer: float,
    previous_temperature_vector: Optional[numpy.ndarray],
    pvt_collector: pvt.PVT,
    resolution: Optional[int],
    element: Element,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing the htf equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    if operating_mode.dynamic:
        bulk_water_internal_energy: float = (
            numpy.pi
            * (pvt_collector.absorber.inner_pipe_diameter / 2) ** 2  # [m^2]
            * element.length  # [m]
            * density_of_water(
                best_guess_temperature_vector[
                    index_handler.index_from_pipe_coordinates(
                        number_of_pipes,
                        number_of_x_elements,
                        element.pipe_index,  # type: ignore
                        pvt_collector,
                        TemperatureName.htf,
                        element.y_index,
                    )
                ]
            )  # [kg/m^3]
            * pvt_collector.absorber.htf_heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )  # [W/K]
    else:
        bulk_water_internal_energy = 0

    fluid_input_output_transfer_term = (
        pvt_collector.absorber.mass_flow_rate  # [kg/s]
        * pvt_collector.absorber.htf_heat_capacity  # [J/kg*K]
        / pvt_collector.absorber.number_of_pipes
    )  # [W/K]

    # Compute the T_f(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_elements,
            element.pipe_index,  # type: ignore
            pvt_collector,
            TemperatureName.htf,
            element.y_index,
        )
    ] = (
        bulk_water_internal_energy + pipe_to_bulk_water_heat_transfer
    )

    # Compute the T_f,in(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_elements,
            element.pipe_index,  # type: ignore
            pvt_collector,
            TemperatureName.htf_in,
            element.y_index,
        )
    ] = (
        -1 * fluid_input_output_transfer_term
    )

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
    ] = fluid_input_output_transfer_term

    # Compute the T_P(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_elements,
            element.pipe_index,  # type: ignore
            pvt_collector,
            TemperatureName.pipe,
            element.y_index,
        )
    ] = (
        -1 * pipe_to_bulk_water_heat_transfer
    )

    # Compute the resultant vector value.
    resultant_vector_value = 0

    if operating_mode.dynamic:
        resultant_vector_value += (
            # Internal heat change.
            bulk_water_internal_energy
            * previous_temperature_vector[  # type: ignore
                index_handler.index_from_pipe_coordinates(
                    number_of_pipes,
                    number_of_x_elements,
                    element.pipe_index,  # type: ignore
                    pvt_collector,
                    TemperatureName.htf,
                    element.y_index,
                )
            ]
        )

    return row_equation, resultant_vector_value
