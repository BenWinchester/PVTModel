#!/usr/bin/python3.7
########################################################################################
# convergent_solver.py - Module for recursively calculating the matrix until convergence
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
Module for recursively calculating the matrix until convergence

This module recursively calls the matrix module until convergence is reached.

"""

import datetime
import logging

from typing import List, Optional, Union

import numpy

from scipy import linalg  # type: ignore

from . import exchanger, tank
from .matrix import matrix
from .pvt_collector import pvt

from ..__utils__ import BColours, OperatingMode, ProgrammerJudgementFault
from .__utils__ import (
    DivergentSolutionError,
    PVT_SYSTEM_MODEL_LOGGER_NAME,
    WeatherConditions,
)
from .constants import (
    CONVERGENT_SOLUTION_PRECISION,
    MAXIMUM_RECURSION_DEPTH,
    WARN_RECURSION_DEPTH,
    ZERO_CELCIUS_OFFSET,
)

__all__ = ("solve_temperature_vector_convergence_method",)


def _calculate_vector_difference(
    first_vector: Union[List[float], numpy.ndarray],
    second_vector: Union[List[float], numpy.ndarray],
) -> float:
    """
    Computes a measure of the difference between two vectors.

    :param first_vector:
        The first vector.

    :param second_vector:
        The second vector.

    :return:
        A measure of the difference between the two vectors.

    """

    # Compute the gross difference between the vectors.
    try:
        diff_vector: List[float] = [
            first_vector[index] - second_vector[index]
            for index in range(len(first_vector))
        ]
    except ValueError as e:
        raise ProgrammerJudgementFault(
            "Atempt was made to compute the difference between two vectors of "
            f"different sizes: {str(e)}"
        ) from None

    # Take the sum of the values in this vector to avoid sign issues and return the sum.
    return sum({abs(value) for value in diff_vector})


def solve_temperature_vector_convergence_method(
    *,
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    number_of_y_elements: int,
    operating_mode: OperatingMode,
    pvt_collector: pvt.PVT,
    run_one_temperature_vector: Union[List[float], numpy.ndarray],
    weather_conditions: WeatherConditions,
    convergence_run_number: int = 0,
    run_one_temperature_difference: float = 5 * ZERO_CELCIUS_OFFSET ** 2,
    collector_input_temperature: Optional[float] = None,
    current_hot_water_load: Optional[float] = None,
    heat_exchanger: Optional[exchanger.Exchanger] = None,
    hot_water_tank: Optional[tank.Tank] = None,
    next_date_and_time: Optional[datetime.datetime] = None,
    previous_run_temperature_vector: Optional[Union[List[float], numpy.ndarray]] = None,
    resolution: Optional[int] = None,
) -> Union[List[float], numpy.ndarray]:
    """
    Itteratively solves for the temperature vector to find a convergent solution.

    The method used to compute the temperatures at the next time step involves
    approximating a non-linear temperature dependance using a best-guess set of
    temperatures for the temperatures at the next time step.

    The "best guess" temperature vector is fed into the matrix solver method, and, from
    this, approximations for the non-linear terms are computed. This then returns a
    "best guess" matrix, from which the temperatures at the next time step are computed.

    Whether the solution is converging, or has converged, is determined.
        - If the solution has converged, the temperature vector computed is returned;
        - If the solution has diverged, an error is raised;
        - If the solution is converging, but has not yet converged, then the function
          runs through again.

    :param collector_input_temperature:
        If set, this governs the input water temperature of the feedwater entering the
        absorber.

    :param current_hot_water_load:
        The current hot-water load placed on the system, measured in kilograms per
        second.

    :param heat_exchanger:
        The heat exchanger being modelled in the system.

    :param hot_water_tank:
        A :class:`tank.Tank` instance representing the hot-water tank being modelled in
        the system.

    :param logger:
        The logger for the module run.

    :param next_date_and_time:
        The date and time at the time step being solved.

    :param number_of_pipes:
        The nmber of pipes on the PVT panel.

    :param number_of_temperatures:
        The number of unique temperature values to determine for the model.

    :param number_of_x_elements:
        The number of elements in one "row" of the panel, i.e., which have the same y
        coordinate

    :param number_of_y_elements:
        The number of elements in one "column" of the panel, i.e., which have the same x
        coordiante.

    :param operating_mode:
        The operating mode for the run.

    :param previous_run_temperature_vector:
        The temperatures at the previous time step.

    :param pvt_collector:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The resolution used for the run, measured in seconds.

    :param run_one_temperature_vector:
        The temperature vector at the last run of the convergent solver.

    :param weather_conditions:
        The weather conditions at the time step being computed.

    :param run_one_temperature_difference:
        The temperature difference between the two vectors when the function was
        previously run.

    :return:
        The temperatures at the next time step.

    :raises: DivergentSolutionError
        Raised if the solution starts to diverge.

    """

    logger.info(
        "Date and time: %s; Run number: %s: Beginning convergent calculation.",
        next_date_and_time.strftime("%d/%m/%Y %H:%M:%S")
        if next_date_and_time is not None
        else "[time-independent]",
        convergence_run_number,
    )

    if operating_mode.coupled:
        coefficient_matrix, resultant_vector = matrix.calculate_matrix_equation(
            best_guess_temperature_vector=run_one_temperature_vector,
            heat_exchanger=heat_exchanger,
            hot_water_load=current_hot_water_load,
            hot_water_tank=hot_water_tank,
            logger=logger,
            number_of_pipes=number_of_pipes,
            number_of_temperatures=number_of_temperatures,
            number_of_x_elements=number_of_x_elements,
            number_of_y_elements=number_of_y_elements,
            operating_mode=operating_mode,
            previous_temperature_vector=numpy.asarray(previous_run_temperature_vector),
            pvt_collector=pvt_collector,
            resolution=resolution,
            weather_conditions=weather_conditions,
        )
    elif operating_mode.steady_state and operating_mode.decoupled:
        coefficient_matrix, resultant_vector = matrix.calculate_matrix_equation(
            best_guess_temperature_vector=run_one_temperature_vector,
            collector_input_temperature=collector_input_temperature,
            logger=logger,
            number_of_pipes=number_of_pipes,
            number_of_temperatures=number_of_temperatures,
            number_of_x_elements=number_of_x_elements,
            number_of_y_elements=number_of_y_elements,
            operating_mode=operating_mode,
            pvt_collector=pvt_collector,
            resolution=resolution,
            weather_conditions=weather_conditions,
        )
    elif operating_mode.dynamic and operating_mode.decoupled:
        coefficient_matrix, resultant_vector = matrix.calculate_matrix_equation(
            best_guess_temperature_vector=run_one_temperature_vector,
            collector_input_temperature=collector_input_temperature,
            logger=logger,
            number_of_pipes=number_of_pipes,
            number_of_temperatures=number_of_temperatures,
            number_of_x_elements=number_of_x_elements,
            number_of_y_elements=number_of_y_elements,
            operating_mode=operating_mode,
            previous_temperature_vector=numpy.asarray(previous_run_temperature_vector),
            pvt_collector=pvt_collector,
            resolution=resolution,
            weather_conditions=weather_conditions,
        )

    logger.debug(
        "Matrix equation computed.\nA =\n%s\nB =\n%s",
        str("\n".join([",".join([str(e) for e in row]) for row in coefficient_matrix])),
        str(resultant_vector),
    )

    try:
        run_two_output = linalg.solve(a=coefficient_matrix, b=resultant_vector)
    except ValueError:
        logger.error(
            "%sMatrix has dimensions %s and is not square.%s",
            BColours.FAIL,
            coefficient_matrix.shape,
            BColours.ENDC,
        )
        raise
    # run_two_output = linalg.lstsq(a=coefficient_matrix, b=resultant_vector)
    run_two_temperature_vector: Union[List[float], numpy.ndarray] = numpy.asarray(  # type: ignore
        [run_two_output[index][0] for index in range(len(run_two_output))]
    )
    # run_two_temperature_vector = run_two_output[0].transpose()[0]

    logger.debug(
        "Date and time: %s; Run number: %s: "
        "Temperatures successfully computed. Temperature vector: T = %s",
        next_date_and_time.strftime("%d/%m/%Y %H:%M:%S")
        if next_date_and_time is not None
        else "[time-independent]",
        convergence_run_number,
        run_two_temperature_vector,
    )

    run_two_temperature_difference = _calculate_vector_difference(
        run_one_temperature_vector, run_two_temperature_vector
    )

    # If the solution has converged, return the temperature vector.
    if run_two_temperature_difference < (10 ** -CONVERGENT_SOLUTION_PRECISION):
        logger.debug(
            "Date and time: %s; Run number: %s: Convergent solution found. "
            "Convergent difference: %s",
            next_date_and_time.strftime("%d/%m/%Y %H:%M:%S")
            if next_date_and_time is not None
            else "[time-independent]",
            convergence_run_number,
            run_two_temperature_difference,
        )
        return run_two_temperature_vector

    logger.debug(
        "Date and time: %s; Run number: %s: Solution not convergent. "
        "Convergent difference: %s",
        next_date_and_time.strftime("%d/%m/%Y %H:%M:%S")
        if next_date_and_time is not None
        else "[time-independent]",
        convergence_run_number,
        run_two_temperature_difference,
    )

    if convergence_run_number >= WARN_RECURSION_DEPTH:
        logger.warn(
            "%sRun depth greater than, or equal, to 10 reached when attempting a "
            "convergent solution.%s",
            BColours.FAIL,
            BColours.ENDC,
        )

    if convergence_run_number >= MAXIMUM_RECURSION_DEPTH:
        logger.error(
            "%sRun depth exceeded maximum recursion depth of %s reached when "
            "attempting a convergent solution.%s",
            BColours.FAIL,
            MAXIMUM_RECURSION_DEPTH,
            BColours.ENDC,
        )
        raise RecursionError(
            BColours.FAIL
            + "Maximum recursion level specified by the constants module reached."
            + BColours.ENDC
        )

    # If the solution has diverged, raise an Exception.
    if round(run_two_temperature_difference, CONVERGENT_SOLUTION_PRECISION) > round(
        run_one_temperature_difference, CONVERGENT_SOLUTION_PRECISION
    ):
        if convergence_run_number > 2:
            logger.error(
                "The temperature solutions at the next time step diverged. "
                "See %s for more details.",
                PVT_SYSTEM_MODEL_LOGGER_NAME,
            )
            logger.info(
                "Local variables at the time of the dump:\n%s",
                "\n".join([f"{key}: {value}" for key, value in locals().items()]),
            )
            raise DivergentSolutionError(
                convergence_run_number,
                run_one_temperature_difference,
                run_one_temperature_vector,
                run_two_temperature_difference,
                run_two_temperature_vector,
            )
        logger.debug("Continuing as fewer than two runs have been attempted...")

    # Otherwise, continue to solve until the prevision is reached.
    return solve_temperature_vector_convergence_method(
        collector_input_temperature=collector_input_temperature,
        current_hot_water_load=current_hot_water_load,
        heat_exchanger=heat_exchanger,
        hot_water_tank=hot_water_tank,
        logger=logger,
        next_date_and_time=next_date_and_time,
        number_of_pipes=number_of_pipes,
        number_of_temperatures=number_of_temperatures,
        number_of_x_elements=number_of_x_elements,
        number_of_y_elements=number_of_y_elements,
        operating_mode=operating_mode,
        previous_run_temperature_vector=previous_run_temperature_vector,
        pvt_collector=pvt_collector,
        resolution=resolution,
        run_one_temperature_vector=run_two_temperature_vector,
        weather_conditions=weather_conditions,
        convergence_run_number=convergence_run_number + 1,
        run_one_temperature_difference=run_two_temperature_difference,
    )
