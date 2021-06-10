#!/usr/bin/python3.7
########################################################################################
# tank.py - The tank module for the matrix component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The tank module for the matrix component.

This module computes and returns the equation(s) associated with the tank layer of the
PV-T collector for the matrix component.

The model works by arranging the system of differential equations as a matrix equation
such that the temperatures at each time step can be computed based on the coefficients
of the matrix which are computed based on the temperatures of the various components at
the previous time step, as well as various Physical and fundamental constants.

"""


import logging

from typing import List, Optional, Tuple, Union

import numpy

from .. import exchanger, index_handler, tank
from ..pvt_collector import pvt

from ...__utils__ import (
    TemperatureName,
)
from ..__utils__ import WeatherConditions
from ..constants import HEAT_CAPACITY_OF_WATER

__all__ = ("calculate_tank_continuity_equation", "calculate_tank_equation")


def calculate_tank_continuity_equation(
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    heat_exchanger: exchanger.Exchanger,
    number_of_temperatures: int,
    pvt_collector: pvt.PVT,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing the tank continuity.

    The HTF flowing through the heat exchanger in the hot-water tank needs to have its
    output temperature computed. The continuity of this fluid is expressed here.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    # If the flow is through the tank heat exchanger:
    if (
        best_guess_temperature_vector[
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.tank_in,
            )
        ]
        > best_guess_temperature_vector[
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.tank,
            )
        ]
    ):
        row_equation[
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.tank,
            )
        ] = (
            -1 * heat_exchanger.efficiency
        )
        row_equation[
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.tank_in,
            )
        ] = (
            heat_exchanger.efficiency - 1
        )
        row_equation[
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.tank_out,
            )
        ] = 1
        return row_equation, 0

    # Otherwise, the flow is diverted back into the absorber.
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.tank_in,
        )
    ] = -1
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.tank_out,
        )
    ] = 1
    return row_equation, 0


def calculate_tank_equation(
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    heat_exchanger: exchanger.Exchanger,
    hot_water_load: float,
    hot_water_tank: tank.Tank,
    logger: logging.Logger,
    number_of_temperatures: int,
    previous_temperature_vector: Optional[numpy.ndarray],
    pvt_collector: pvt.PVT,
    resolution: Optional[int],
    weather_conditions: WeatherConditions,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing the tank equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    logger.debug("Beginning calculation of Tank equation")

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    tank_internal_energy = (
        hot_water_tank.mass  # [kg]
        * HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        / resolution  # type: ignore  # [s]
    )
    logger.debug("Tank internal energy term: %s W/K", tank_internal_energy)

    hot_water_load_term = hot_water_load * HEAT_CAPACITY_OF_WATER  # [kg/s]  # [J/kg*K]
    logger.debug("Tank hot-water load term: %s W/K", hot_water_load_term)

    heat_loss_term = (
        hot_water_tank.heat_loss_coefficient * hot_water_tank.area  # [W/m^2*K]  # [m^2]
    )
    logger.debug("Tank heat-loss term: %s W/K", heat_loss_term)

    heat_input_term = (
        (
            pvt_collector.absorber.mass_flow_rate  # [kg/s]
            * pvt_collector.absorber.htf_heat_capacity  # [J/kg*K]
            * heat_exchanger.efficiency
        )
        if best_guess_temperature_vector[
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.tank_in,
            )
        ]
        > best_guess_temperature_vector[
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.tank,
            )
        ]
        else 0
    )
    logger.debug("Tank heat-input term: %s W/K", heat_input_term)

    # Compute the T_t term
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.tank,
        )
    ] = (
        tank_internal_energy + hot_water_load_term + heat_loss_term + heat_input_term
    )

    # Compute the T_c,out term
    row_equation[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.tank_in,
        )
    ] = (
        -1 * heat_input_term
    )

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Internal heat change
        tank_internal_energy  # [W/K]
        * previous_temperature_vector[  # type: ignore
            index_handler.index_from_temperature_name(
                pvt_collector,
                TemperatureName.tank,
            )
        ]  # [K]
        # Hot-water load.
        + hot_water_load_term  # [W/K]
        * weather_conditions.mains_water_temperature  # [K]
        # Heat loss
        + heat_loss_term * weather_conditions.ambient_tank_temperature  # [W/K]  # [K]
    )

    return row_equation, resultant_vector_value
