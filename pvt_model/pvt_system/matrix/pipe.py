#!/usr/bin/python3.7
########################################################################################
# pipe.py - The pipe module for the matrix component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The pipe module for the matrix component.

This module computes and returns the equation(s) associated with the pipe layer of the
PV-T collector for the matrix component.

The model works by arranging the system of differential equations as a matrix equation
such that the temperatures at each time step can be computed based on the coefficients
of the matrix which are computed based on the temperatures of the various components at
the previous time step, as well as various Physical and fundamental constants.

"""


import logging

from typing import List, Optional, Tuple, Union

import numpy

from .. import index_handler
from ..pvt_collector import pvt

from ...__utils__ import (
    OperatingMode,
    TemperatureName,
)
from ..__utils__ import WeatherConditions
from ..pvt_collector.element import Element
from ..pvt_collector.physics_utils import insulation_thermal_resistance

__all__ = "calculate_pipe_equation"


def calculate_pipe_equation(
    absorber_to_pipe_conduction: float,
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    operating_mode: OperatingMode,
    pipe_to_htf_heat_transfer: float,
    previous_temperature_vector: Optional[numpy.ndarray],
    pvt_collector: pvt.PVT,
    resolution: Optional[int],
    element: Element,
    weather_conditions: WeatherConditions,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing the pipe equation.

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
        pipe_internal_heat_change: float = (
            numpy.pi
            * (
                (pvt_collector.absorber.outer_pipe_diameter / 2) ** 2  # [m^2]
                - (pvt_collector.absorber.inner_pipe_diameter / 2) ** 2  # [m^2]
            )
            * element.length  # [m]
            * pvt_collector.absorber.pipe_density  # [kg/m^3]
            * pvt_collector.absorber.heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )
    else:
        pipe_internal_heat_change = 0

    pipe_to_surroundings_losses = (
        numpy.pi
        * (pvt_collector.absorber.outer_pipe_diameter / 2)  # [m]
        * element.length  # [m]
        / insulation_thermal_resistance(
            best_guess_temperature_vector,
            pvt_collector,
            index_handler.index_from_pipe_coordinates(
                number_of_pipes,
                number_of_x_elements,
                element.pipe_index,  # type: ignore
                pvt_collector,
                TemperatureName.pipe,
                element.y_index,
            ),
            weather_conditions,
        )  # [K*m^2/W]
    )

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
        pipe_internal_heat_change  # [W/K]
        + absorber_to_pipe_conduction  # [W/K]
        + pipe_to_htf_heat_transfer  # [W/K]
        + pipe_to_surroundings_losses
    )

    # Compute the T_A(i, j) term.
    row_equation[
        index_handler.index_from_element_coordinates(
            number_of_x_elements,
            pvt_collector,
            TemperatureName.absorber,
            element.x_index,
            element.y_index,
        )
    ] = -1 * (absorber_to_pipe_conduction)

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
    ] = -1 * (pipe_to_htf_heat_transfer)

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Ambient heat loss.
        pipe_to_surroundings_losses
        * weather_conditions.ambient_temperature  # [W]
    )

    if operating_mode.dynamic:
        resultant_vector_value += (
            # Internal heat change.
            pipe_internal_heat_change
            * previous_temperature_vector[  # type: ignore
                index_handler.index_from_pipe_coordinates(
                    number_of_pipes,
                    number_of_x_elements,
                    element.pipe_index,  # type: ignore
                    pvt_collector,
                    TemperatureName.pipe,
                    element.y_index,
                )
            ]
        )

    logger.debug(
        "Rough Pipe Temperature estimate: %s K.",
        int(
            resultant_vector_value
            / (pipe_to_surroundings_losses + pipe_internal_heat_change)
        ),
    )

    return row_equation, resultant_vector_value
