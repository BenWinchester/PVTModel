#!/usr/bin/python3.7
########################################################################################
# upper_glass.py - The upper_glass module for the matrix component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The upper_glass module for the matrix component.

This module computes and returns the equation(s) associated with the upper_glass layer
of the PV-T collector for the matrix component.

The model works by arranging the system of differential equations as a matrix equation
such that the temperatures at each time step can be computed based on the coefficients
of the matrix which are computed based on the temperatures of the various components at
the previous time step, as well as various Physical and fundamental constants.

"""


import logging

from typing import List, Optional, Tuple, Union

import numpy

from .. import index_handler, physics_utils
from ..pvt_collector import pvt

from ...__utils__ import (
    BColours,
    OperatingMode,
    ProgrammerJudgementFault,
    TemperatureName,
)
from ..__utils__ import WeatherConditions
from ..pvt_collector.element import Element, ElementCoordinates


__all__ = ("calculate_upper_glass_equation",)


def calculate_upper_glass_equation(  # pylint: disable=too-many-branches
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    logger: logging.Logger,
    number_of_temperatures: int,
    number_of_x_elements: int,
    number_of_y_elements: int,
    operating_mode: OperatingMode,
    previous_temperature_vector: Optional[numpy.ndarray],
    pvt_collector: pvt.PVT,
    resolution: Optional[int],
    element: Element,
    upper_glass_downward_conduction: float,
    upper_glass_downward_radiation: float,
    weather_conditions: WeatherConditions,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing the upper_glass equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    logger.debug(
        "Beginning calculation of upper_glass equation for element %s.",
        element.coordinates,
    )

    if pvt_collector.upper_glass is None:
        raise ProgrammerJudgementFault(
            "{}Element {} has a upper_glass layer but no upper_glass data supplied ".format(
                BColours.FAIL, element
            )
            + "in the PV-T data file.{}".format(BColours.ENDC)
        )

    if operating_mode.dynamic:
        upper_glass_internal_energy: float = (
            element.width  # [m]
            * element.length  # [m]
            * pvt_collector.upper_glass.thickness  # [m]
            * pvt_collector.upper_glass.density  # [kg/m^3]
            * pvt_collector.upper_glass.heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )
    else:
        upper_glass_internal_energy = 0
    logger.debug(
        "Upper glass internal energy term: %s W/K", upper_glass_internal_energy
    )

    # Compute the positive conductive term based on the next element along.
    positive_x_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index + 1, element.y_index)
    )
    if positive_x_element is not None:
        positive_x_wise_conduction: float = (
            pvt_collector.upper_glass.conductivity  # [W/m*K]
            * pvt_collector.upper_glass.thickness  # [m]
            * element.length  # [m]
            / (0.5 * (element.width + positive_x_element.width))  # [m]
        )
    else:
        positive_x_wise_conduction = 0
    logger.debug(
        "Positive upper_glass x-wise conduction term: %s W/K",
        positive_x_wise_conduction,
    )

    # Compute the positive conductive term based on the next element along.
    negative_x_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index - 1, element.y_index)
    )
    if negative_x_element is not None:
        negative_x_wise_conduction: float = (
            pvt_collector.upper_glass.conductivity  # [W/m*K]
            * pvt_collector.upper_glass.thickness  # [m]
            * element.length  # [m]
            / (0.5 * (element.width + negative_x_element.width))  # [m]
        )
    else:
        negative_x_wise_conduction = 0
    logger.debug(
        "Negative upper_glass x-wise conduction term: %s W/K",
        negative_x_wise_conduction,
    )

    # Compute the overall x-wise conduction term.
    x_wise_conduction = positive_x_wise_conduction + negative_x_wise_conduction
    logger.debug("Glass x-wise conduction term: %s W/K", x_wise_conduction)

    # Compute the positive conductive term based on the next element along.
    positive_y_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index, element.y_index + 1)
    )
    if positive_y_element is not None:
        positive_y_wise_conduction: float = (
            pvt_collector.upper_glass.conductivity  # [W/m*K]
            * pvt_collector.upper_glass.thickness  # [m]
            * element.width  # [m]
            / (0.5 * (element.length + positive_y_element.length))  # [m]
        )
    else:
        positive_y_wise_conduction = 0
    logger.debug(
        "Positive upper_glass y-wise conduction term: %s W/K",
        positive_y_wise_conduction,
    )

    # Compute the positive conductive term based on the next element along.
    negative_y_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index, element.y_index - 1)
    )
    if negative_y_element is not None:
        negative_y_wise_conduction: float = (
            pvt_collector.upper_glass.conductivity  # [W/m*K]
            * pvt_collector.upper_glass.thickness  # [m]
            * element.width  # [m]
            / (0.5 * (element.length + negative_y_element.length))  # [m]
        )
    else:
        negative_y_wise_conduction = 0
    logger.debug(
        "Negative upper_glass y-wise conduction term: %s W/K",
        negative_y_wise_conduction,
    )

    # Compute the overall y-wise conduction term.
    y_wise_conduction = positive_y_wise_conduction + negative_y_wise_conduction
    logger.debug("Glass y-wise conduction term: %s W/K", y_wise_conduction)

    (
        upper_glass_to_air_conduction,
        upper_glass_to_sky_radiation,
    ) = physics_utils.upward_loss_terms(
        best_guess_temperature_vector,
        pvt_collector,
        element,
        pvt_collector.upper_glass.emissivity,
        index_handler.index_from_element_coordinates(
            number_of_x_elements,
            pvt_collector,
            TemperatureName.upper_glass,
            element.x_index,
            element.y_index,
        ),
        weather_conditions,
    )
    logger.debug("Upper glass to air conduction %s W/K", upper_glass_to_air_conduction)
    logger.debug("Upper glass to sky radiation %s W/K", upper_glass_to_sky_radiation)

    # Compute the T_ug(i, j) term
    row_equation[
        index_handler.index_from_element_coordinates(
            number_of_x_elements,
            pvt_collector,
            TemperatureName.upper_glass,
            element.x_index,
            element.y_index,
        )
    ] = (
        upper_glass_internal_energy
        + x_wise_conduction
        + y_wise_conduction
        + upper_glass_to_air_conduction
        + upper_glass_to_sky_radiation
        + upper_glass_downward_conduction
        + upper_glass_downward_radiation
    )

    # Compute the T_ug(i+1, j) term provided that that element exists.
    if element.x_index + 1 < number_of_x_elements:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.upper_glass,
                element.x_index + 1,
                element.y_index,
            )
        ] = (
            -1 * positive_x_wise_conduction
        )

    # Compute the T_ug(i-1, j) term provided that that element exists.
    if element.x_index > 0:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.upper_glass,
                element.x_index - 1,
                element.y_index,
            )
        ] = (
            -1 * negative_x_wise_conduction
        )

    # Compute the T_ug(i, j+1) term provided that that element exists.
    if element.y_index + 1 < number_of_y_elements:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.upper_glass,
                element.x_index,
                element.y_index + 1,
            )
        ] = (
            -1 * positive_y_wise_conduction
        )

    # Compute the T_ug(i, j-1) term provided that that element exists.
    if element.y_index > 0:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.upper_glass,
                element.x_index,
                element.y_index - 1,
            )
        ] = (
            -1 * negative_y_wise_conduction
        )

    # Compute the T_g(i, j) term.
    row_equation[
        index_handler.index_from_element_coordinates(
            number_of_x_elements,
            pvt_collector,
            TemperatureName.glass,
            element.x_index,
            element.y_index,
        )
    ] = -1 * (upper_glass_downward_conduction + upper_glass_downward_radiation)

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Ambient temperature term.
        upper_glass_to_air_conduction * weather_conditions.ambient_temperature  # [W]
        # Sky temperature term.
        + upper_glass_to_sky_radiation * weather_conditions.sky_temperature  # [W]
        # Solar absorption term.
        + element.width  # [m]
        * element.length  # [m]
        * pvt_collector.upper_glass_transmissivity_absorptivity_product
        * weather_conditions.irradiance  # [W/m^2]
    )

    if operating_mode.dynamic:
        resultant_vector_value += (
            # Previous upper_glass temperature term.
            upper_glass_internal_energy  # [W/K]
            * previous_temperature_vector[  # type: ignore
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    TemperatureName.upper_glass,
                    element.x_index,
                    element.y_index,
                )
            ]  # [K]
        )

        logger.debug(
            "Rough Glass Temperature estimate: %s K.",
            int(
                resultant_vector_value
                / (
                    upper_glass_to_air_conduction
                    + upper_glass_to_sky_radiation
                    + upper_glass_internal_energy
                )
            ),
        )

    return row_equation, resultant_vector_value
