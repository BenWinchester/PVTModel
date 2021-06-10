#!/usr/bin/python3.7
########################################################################################
# pv.py - The pv module for the matrix component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The pv module for the matrix component.

This module computes and returns the equation(s) associated with the PV layer of the
PV-T collector for the matrix component.

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

from ...__utils__ import OperatingMode, TemperatureName
from ..__utils__ import WeatherConditions
from ..efficiency import electrical_efficiency
from ..pvt_collector.element import Element, ElementCoordinates

__all__ = ("calculate_pv_equation",)


def calculate_pv_equation(  # pylint: disable=too-many-branches
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    logger: logging.Logger,
    number_of_temperatures: int,
    number_of_x_elements: int,
    number_of_y_elements: int,
    operating_mode: OperatingMode,
    previous_temperature_vector: Optional[numpy.ndarray],
    pv_to_absorber_conduction: float,
    pv_to_glass_conduction: float,
    pv_to_glass_radiation: float,
    pvt_collector: pvt.PVT,
    resolution: Optional[int],
    element: Element,
    weather_conditions: WeatherConditions,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing the pv equation.

    :param logger:
        The logger used in the run.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    # if weather_conditions.irradiance > 0:
    #     import pdb

    #     pdb.set_trace(header=f"T_PV{element.coordinates}")

    logger.debug(
        "Beginning calculation of PV equation for element %s.", element.coordinates
    )

    if operating_mode.dynamic:
        pv_internal_energy: float = (
            element.width  # [m]
            * element.length  # [m]
            * pvt_collector.pv.thickness  # [m]
            * pvt_collector.pv.density  # [kg/m^3]
            * pvt_collector.pv.heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )
    else:
        pv_internal_energy = 0
    logger.debug("PV internal energy term: %s W/K", pv_internal_energy)

    # Compute the upward environmental losses.
    if not element.glass:
        (pv_to_air_conduction, pv_to_sky_radiation,) = physics_utils.upward_loss_terms(
            best_guess_temperature_vector,
            pvt_collector,
            element,
            pvt_collector.pv.emissivity,
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.pv,
                element.x_index,
                element.y_index,
            ),
            weather_conditions,
        )
        logger.debug("PV to air conduction %s W/K", pv_to_air_conduction)
        logger.debug("PV to sky radiation %s W/K", pv_to_sky_radiation)
    else:
        pv_to_air_conduction = 0
        pv_to_sky_radiation = 0

    # Compute the positive conductive term based on the next element along.
    positive_x_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index + 1, element.y_index)
    )
    if positive_x_element is not None:
        positive_x_wise_conduction: float = (
            pvt_collector.pv.conductivity  # [W/m*K]
            * pvt_collector.pv.thickness  # [m]
            * element.length  # [m]
            / (0.5 * (element.width + positive_x_element.width))  # [m]
        )
    else:
        positive_x_wise_conduction = 0
    logger.debug(
        "Positive PV x-wise conduction term: %s W/K", positive_x_wise_conduction
    )

    # Compute the positive conductive term based on the next element along.
    negative_x_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index - 1, element.y_index)
    )
    if negative_x_element is not None:
        negative_x_wise_conduction: float = (
            pvt_collector.pv.conductivity  # [W/m*K]
            * pvt_collector.pv.thickness  # [m]
            * element.length  # [m]
            / (0.5 * (element.width + negative_x_element.width))  # [m]
        )
    else:
        negative_x_wise_conduction = 0
    logger.debug(
        "Negative PV x-wise conduction term: %s W/K", negative_x_wise_conduction
    )

    # Compute the overall x-wise conduction term.
    x_wise_conduction = positive_x_wise_conduction + negative_x_wise_conduction
    logger.debug("PV x-wise conduction term: %s W/K", x_wise_conduction)

    # Compute the positive conductive term based on the next element along.
    positive_y_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index, element.y_index + 1)
    )
    if positive_y_element is not None:
        positive_y_wise_conduction: float = (
            pvt_collector.pv.conductivity  # [W/m*K]
            * pvt_collector.pv.thickness  # [m]
            * element.width  # [m]
            / (0.5 * (element.length + positive_y_element.length))  # [m]
        )
    else:
        positive_y_wise_conduction = 0
    logger.debug(
        "Positive PV y-wise conduction term: %s W/K", positive_y_wise_conduction
    )

    # Compute the positive conductive term based on the next element along.
    negative_y_element = pvt_collector.elements.get(
        ElementCoordinates(element.x_index, element.y_index - 1)
    )
    if negative_y_element is not None:
        negative_y_wise_conduction: float = (
            pvt_collector.pv.conductivity  # [W/m*K]
            * pvt_collector.pv.thickness  # [m]
            * element.width  # [m]
            / (0.5 * (element.length + negative_y_element.length))  # [m]
        )
    else:
        negative_y_wise_conduction = 0
    logger.debug(
        "Negative PV y-wise conduction term: %s W/K", negative_y_wise_conduction
    )

    # Compute the overall y-wise conduction term.
    y_wise_conduction = positive_y_wise_conduction + negative_y_wise_conduction
    logger.debug("PV y-wise conduction term: %s W/K", y_wise_conduction)

    # Compute the T_pv(i, j) term
    row_equation[
        index_handler.index_from_element_coordinates(
            number_of_x_elements,
            pvt_collector,
            TemperatureName.pv,
            element.x_index,
            element.y_index,
        )
    ] = (
        pv_internal_energy
        + x_wise_conduction
        + y_wise_conduction
        + pv_to_air_conduction
        + pv_to_sky_radiation
        + pv_to_glass_radiation
        + pv_to_glass_conduction
        + pv_to_absorber_conduction
    )

    # Compute the T_pv(i+1, j) term provided that that element exists.
    if element.x_index + 1 < number_of_x_elements:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.pv,
                element.x_index + 1,
                element.y_index,
            )
        ] = (
            -1 * positive_x_wise_conduction
        )

    # Compute the T_pv(i-1, j) term provided that that element exists.
    if element.x_index > 0:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.pv,
                element.x_index - 1,
                element.y_index,
            )
        ] = (
            -1 * negative_x_wise_conduction
        )

    # Compute the T_pv(i, j+1) term provided that that element exists.
    if element.y_index + 1 < number_of_y_elements:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.pv,
                element.x_index,
                element.y_index + 1,
            )
        ] = (
            -1 * positive_y_wise_conduction
        )

    # Compute the T_pv(i, j-1) term provided that that element exists.
    if element.y_index > 0:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.pv,
                element.x_index,
                element.y_index - 1,
            )
        ] = (
            -1 * negative_y_wise_conduction
        )

    # Compute the T_g(i, j) term provided that there is a glass layer present.
    if element.glass:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.glass,
                element.x_index,
                element.y_index,
            )
        ] = -1 * (pv_to_glass_conduction + pv_to_glass_radiation)

    # Compute the T_A(i, j) term provided that there is a absorber layer present.
    if element.absorber:
        row_equation[
            index_handler.index_from_element_coordinates(
                number_of_x_elements,
                pvt_collector,
                TemperatureName.absorber,
                element.x_index,
                element.y_index,
            )
        ] = (
            -1 * pv_to_absorber_conduction
        )

    solar_thermal_resultant_vector_absorbtion_term = (
        pvt_collector.pv_transmissivity_absorptivity_product
        * weather_conditions.irradiance  # [W/m^2]
        * element.width  # [m]
        * element.length  # [m]
    ) - (
        weather_conditions.irradiance  # [W/m^2]
        * element.width  # [m]
        * element.length  # [m]
        * electrical_efficiency(
            pvt_collector,
            best_guess_temperature_vector[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    TemperatureName.pv,
                    element.x_index,
                    element.y_index,
                )
            ],
        )
    )
    logger.debug(
        "PV solar thermal resultant vector term: %s W/K",
        solar_thermal_resultant_vector_absorbtion_term,
    )

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Solar thermal absorption term.
        solar_thermal_resultant_vector_absorbtion_term  # [W]
        # Ambient temperature term.
        + pv_to_air_conduction * weather_conditions.ambient_temperature  # [W]
        # Sky temperature term.
        + pv_to_sky_radiation * weather_conditions.sky_temperature  # [W]
    )

    if operating_mode.dynamic:
        resultant_vector_value += (
            # Internal energy change
            pv_internal_energy  # [W/K]
            * previous_temperature_vector[  # type: ignore
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    TemperatureName.pv,
                    element.x_index,
                    element.y_index,
                )
            ]
        )

    return row_equation, resultant_vector_value
