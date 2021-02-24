#!/usr/bin/python3.7
########################################################################################
# matrix.py - The matrix coefficient solver for the PV-T model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The matrix coefficient solver for the PV-T model.

The model works by arranging the system of differential equations as a matrix equation
such that the temperatures at each time step can be computed based on the coefficients
of the matrix which are computed based on the temperatures of the various components at
the previous time step, as well as various Physical and fundamental constants.

The equations in the system can be expressed as the matrix equation
    A * T = B
where
    A is the matrix computed and returned by this function;
    T is a vector containing the various temperatures;
    and B is the "resultant vector," computed elsewhere.

The equations represented by the rows of the matrix are:


The temperatures represented in the vector T are:

"""

from typing import Set, Tuple

import numpy

from . import index
from .pvt_panel import pvt

from ..__utils__ import TemperatureName
from .__utils__ import WeatherConditions

__all__ = ("calculate_matrix_equation",)


####################
# Internal methods #
####################


def _absorber_equation(
    number_of_temperatures: int,
) -> Tuple[numpy.ndarray, Tuple[float, ...]]:
    """
    Returns a matrix row and resultant vector value representing the absorber equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # * Compute the row equation

    # * Compute the resultant vector value.


def _fluid_continuity_equation(
    number_of_temperatures: int,
) -> Tuple[numpy.ndarray, Tuple[float, ...]]:
    """
    Returns a matrix row and resultant vector value representing continuity of the htf.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # * Compute the row equation

    # * Compute the resultant vector value.


def _glass_equation(
    number_of_temperatures: int,
) -> Tuple[numpy.ndarray, Tuple[float, ...]]:
    """
    Returns a matrix row and resultant vector value representing the glass equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # * Compute the row equation

    # * Compute the resultant vector value.


def _htf_equation(
    number_of_temperatures: int,
) -> Tuple[numpy.ndarray, Tuple[float, ...]]:
    """
    Returns a matrix row and resultant vector value representing the glass equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # * Compute the row equation

    # * Compute the resultant vector value.


def _pipe_equation(
    number_of_temperatures: int,
) -> Tuple[numpy.ndarray, Tuple[float, ...]]:
    """
    Returns a matrix row and resultant vector value representing the pipe equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # * Compute the row equation

    # * Compute the resultant vector value.


def _pv_equation(
    number_of_temperatures: int,
) -> Tuple[numpy.ndarray, Tuple[float, ...]]:
    """
    Returns a matrix row and resultant vector value representing the pv equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # * Compute the row equation

    # * Compute the resultant vector value.


def _system_continuity_equations(
    number_of_temperatures: int,
) -> Set[Tuple[numpy.ndarray, Tuple[float, ...]]]:
    """
    Returns matrix rows and resultant vector values representing system continuities.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `set` of `tuple`s containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # * Compute the row equation

    # * Compute the resultant vector value.


def _tank_equation(
    number_of_temperatures: int,
) -> Tuple[numpy.ndarray, Tuple[float, ...]]:
    """
    Returns a matrix row and resultant vector value representing the tank equation.

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `tuple` containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    # * Compute the row equation

    # * Compute the resultant vector value.


##################
# Public methods #
##################


def calculate_matrix_equation() -> Tuple[numpy.ndarray, Tuple[float, ...]]:
    """
    Calculates and returns both the matrix and resultant vector for the matrix equation.

    :return:
        A `tuple` containing both the matrix "A" and resultant vector "B" for the matrix
        equation representing the temperature equations.

    """

    # * Set up an index for tracking the equation number.

    # * Instantiate an empty matrix and array based on the number of temperatures
    # * present.

    # * Determine the number of temperatures being modelled.

    # * Iterate through and generate...

    # * the glass equations,

    # * the pv equations,

    # * the absorber-layer eqations,

    # * and the pipe equations;

    # * the htf equations,

    # * htf input equations,

    # * and htf output equations,

    # * and the various continuity equations,

    # * along with the tank equation.
