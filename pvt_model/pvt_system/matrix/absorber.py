#!/usr/bin/python3.7
########################################################################################
# absorber.py - The absorber module for the matrix component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The absorber module for the matrix component.

This module computes and returns the equation(s) associated with the absorber layer of
the PV-T collector for the matrix component.

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

from ...__utils__ import OperatingMode, ProgrammerJudgementFault, TemperatureName
from ..__utils__ import WeatherConditions
from ..pvt_collector.element import Element, ElementCoordinates
from ..pvt_collector.physics_utils import insulation_thermal_resistance


__all__ = ("calculate_absorber_equation",)


def calculate_absorber_equation(  # pylint: disable=too-many-branches
    absorber_to_pipe_conduction: float,
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    glass_downward_conduction: float,
    glass_downward_radiation: float,
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    number_of_y_elements: int,
    operating_mode: OperatingMode,
    previous_temperature_vector: Optional[numpy.ndarray],
    pv_to_absorber_conduction: float,
    pvt_collector: pvt.PVT,
    resolution: Optional[int],
    element: Element,
    weather_conditions: WeatherConditions,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing the absorber equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    logger.debug(
        "Beginning calculation of absorber equation for element %s.",
        element.coordinates,
    )

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    # import pdb

    # pdb.set_trace(header=f"T_A{element.coordinates}")

    if operating_mode.dynamic:
        collector_internal_energy_change: float = (
            element.width  # [m]
            * element.length  # [m]
            * pvt_collector.absorber.thickness  # [m]
            * pvt_collector.absorber.density  # [kg/m^3]
            * pvt_collector.absorber.heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )
    else:
        collector_internal_energy_change = 0
    logger.debug(
        "Absorber internal energy term: %s W/K", collector_internal_energy_change
    )

    # Compute the positive conductive term based on the next element along.
    positive_x_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index + 1, element.y_index)
    )
    if positive_x_element is not None:
        positive_x_wise_conduction: float = (
            pvt_collector.absorber.conductivity  # [W/m*K]
            * pvt_collector.absorber.thickness  # [m]
            * element.length  # [m]
            / (0.5 * (element.width + positive_x_element.width))  # [m]
        )
    else:
        positive_x_wise_conduction = 0
    logger.debug(
        "Positive absorber x-wise conduction term: %s W/K", positive_x_wise_conduction
    )

    # Compute the positive conductive term based on the next element along.
    negative_x_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index - 1, element.y_index)
    )
    if negative_x_element is not None:
        negative_x_wise_conduction: float = (
            pvt_collector.absorber.conductivity  # [W/m*K]
            * pvt_collector.absorber.thickness  # [m]
            * element.length  # [m]
            / (0.5 * (element.width + negative_x_element.width))  # [m]
        )
    else:
        negative_x_wise_conduction = 0
    logger.debug(
        "Negative absorber x-wise conduction term: %s W/K", negative_x_wise_conduction
    )

    # Compute the overall x-wise conduction term.
    x_wise_conduction = positive_x_wise_conduction + negative_x_wise_conduction
    logger.debug("Absorber x-wise conduction term: %s W/K", x_wise_conduction)

    # Compute the positive conductive term based on the next element along.
    positive_y_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index, element.y_index + 1)
    )
    if positive_y_element is not None:
        positive_y_wise_conduction: float = (
            pvt_collector.absorber.conductivity  # [W/m*K]
            * pvt_collector.absorber.thickness  # [m]
            * element.width  # [m]
            / (0.5 * (element.length + positive_y_element.length))  # [m]
        )
    else:
        positive_y_wise_conduction = 0
    logger.debug(
        "Positive absorber y-wise conduction term: %s W/K", positive_y_wise_conduction
    )

    # Compute the positive conductive term based on the next element along.
    negative_y_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index, element.y_index - 1)
    )
    if negative_y_element is not None:
        negative_y_wise_conduction: float = (
            pvt_collector.absorber.conductivity  # [W/m*K]
            * pvt_collector.absorber.thickness  # [m]
            * element.width  # [m]
            / (0.5 * (element.length + negative_y_element.length))  # [m]
        )
    else:
        negative_y_wise_conduction = 0
    logger.debug(
        "Negative absorber y-wise conduction term: %s W/K", negative_y_wise_conduction
    )

    # Compute the overall y-wise conduction term.
    y_wise_conduction = positive_y_wise_conduction + negative_y_wise_conduction
    logger.debug("Absorber y-wise conduction term: %s W/K", y_wise_conduction)

    absorber_to_insulation_loss = (
        (
            element.width  # [m]
            * element.length  # [m]
            / insulation_thermal_resistance(
                best_guess_temperature_vector,
                pvt_collector,
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    TemperatureName.absorber,
                    element.x_index,
                    element.y_index,
                ),
                weather_conditions,
            )  # [m^2*K/W]
        )
        if not element.pipe
        else 0
    )
    logger.debug(
        "Absorber to insulation loss term: %s W/K", absorber_to_insulation_loss
    )

    # If there are no upper layers, compute the upward loss terms.
    if not element.pv and not element.glass:
        (
            absorber_to_air_conduction,
            absorber_to_sky_radiation,
        ) = physics_utils.upward_loss_terms(
            best_guess_temperature_vector,
            pvt_collector,
            element,
            pvt_collector.absorber.emissivity,
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.absorber,
                element.x_index,
                element.y_index,
            ),
            weather_conditions,
        )
        logger.debug("Absorber to air conduction %s W/K", absorber_to_air_conduction)
        logger.debug("Absorber to sky radiation %s W/K", absorber_to_sky_radiation)
    else:
        absorber_to_air_conduction = 0
        absorber_to_sky_radiation = 0

    # Compute the T_A(i, j) term
    row_equation[
        index_handler.index_from_element_coordinates(
            number_of_x_elements,
            pvt_collector,
            TemperatureName.absorber,
            element.x_index,
            element.y_index,
        )
    ] = (
        collector_internal_energy_change
        + x_wise_conduction
        + y_wise_conduction
        + (pv_to_absorber_conduction if element.pv else 0)
        + absorber_to_pipe_conduction
        + absorber_to_insulation_loss
        + (
            absorber_to_air_conduction + absorber_to_sky_radiation
            if (not element.pv and not element.glass)
            else 0
        )
        + (
            glass_downward_conduction + glass_downward_radiation
            if (element.glass and not element.pv)
            else 0
        )
    )

    # Compute the T_A(i+1, j) term provided that that element exists.
    if element.x_index + 1 < number_of_x_elements:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.absorber,
                element.x_index + 1,
                element.y_index,
            )
        ] = (
            -1 * positive_x_wise_conduction
        )

    # Compute the T_A(i-1, j) term provided that that element exists.
    if element.x_index > 0:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.absorber,
                element.x_index - 1,
                element.y_index,
            )
        ] = (
            -1 * negative_x_wise_conduction
        )

    # Compute the T_A(i, j+1) term provided that that element exists.
    if element.y_index + 1 < number_of_y_elements:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.absorber,
                element.x_index,
                element.y_index + 1,
            )
        ] = (
            -1 * positive_y_wise_conduction
        )

    # Compute the T_A(i, j-1) term provided that that element exists.
    if element.y_index > 0:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.absorber,
                element.x_index,
                element.y_index - 1,
            )
        ] = (
            -1 * negative_y_wise_conduction
        )

    # Compute the T_pv(i, j) term provided that there is a absorber layer present.
    if element.pv:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.pv,
                element.x_index,
                element.y_index,
            )
        ] = (
            -1 * pv_to_absorber_conduction
        )
    elif element.glass:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.glass,
                element.x_index,
                element.y_index,
            )
        ] = -1 * (glass_downward_conduction + glass_downward_radiation)

    # Compute the T_P(pipe_number, j) term provided that there is a pipe present.
    if element.pipe:
        if element.pipe_index is None:
            raise ProgrammerJudgementFault(
                "A element specifies a pipe is present yet no pipe index is supplied. "
                f"Element: {str(element)}"
            )
        try:
            row_equation[
                index_handler.index_from_pipe_coordinates(
                    number_of_pipes,
                    number_of_x_elements,
                    element.pipe_index,
                    pvt_collector,
                    TemperatureName.pipe,
                    element.y_index,
                )
            ] = (
                -1 * absorber_to_pipe_conduction
            )
        except ProgrammerJudgementFault as e:
            raise ProgrammerJudgementFault(
                "Error determining pipe temperature term. "
                f"Likely that pipe index was not specified: {str(e)}"
            ) from None

    # Compute the solar-thermal absorption if relevant.
    if not element.pv:
        solar_thermal_resultant_vector_absorbtion_term = (
            pvt_collector.absorber_transmissivity_absorptivity_product
            * weather_conditions.irradiance  # [W/m^2]
            * element.width  # [m]
            * element.length  # [m]
        )
    else:
        solar_thermal_resultant_vector_absorbtion_term = 0

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Ambient temperature term.
        absorber_to_insulation_loss  # [W/K]
        * weather_conditions.ambient_temperature  # [K]
        # Solar absorption term.
        + solar_thermal_resultant_vector_absorbtion_term  # [W]
        # Heat loss to the air.
        + absorber_to_air_conduction * weather_conditions.ambient_temperature
        # Absorber to sky radiation
        + absorber_to_sky_radiation * weather_conditions.sky_temperature
    )

    if operating_mode.dynamic:
        resultant_vector_value += (
            # Internal heat change term.
            collector_internal_energy_change  # [W/K]
            * previous_temperature_vector[  # type: ignore
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    TemperatureName.absorber,
                    element.x_index,
                    element.y_index,
                )
            ]  # [K]
        )

        try:
            logger.debug(
                "Rough Absorber Temperature estimate: %s K.",
                int(
                    resultant_vector_value
                    / (absorber_to_insulation_loss + collector_internal_energy_change)
                ),
            )
        except ZeroDivisionError:
            logger.debug(
                "Absorber temperature estimate could not be computed due to zero-division."
            )

    return row_equation, resultant_vector_value
