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

from typing import List, Optional, Tuple, Union

import logging
import numpy

from . import exchanger, index_handler, physics_utils, tank
from .pvt_panel import pvt
from .pvt_panel.segment import Segment, SegmentCoordinates

from ..__utils__ import (
    BColours,
    TemperatureName,
    OperatingMode,
    ProgrammerJudgementFault,
)
from .__utils__ import WeatherConditions
from .constants import DENSITY_OF_WATER, HEAT_CAPACITY_OF_WATER
from .physics_utils import (
    radiative_heat_transfer_coefficient,
)
from .pvt_panel.physics_utils import air_gap_resistance, insulation_thermal_resistance

__all__ = ("calculate_matrix_equation",)


####################
# Internal methods #
####################


def _absorber_equation(  # pylint: disable=too-many-branches
    absorber_to_pipe_conduction: float,
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    previous_temperature_vector: Optional[numpy.ndarray],
    pv_to_absorber_conduction: float,
    pvt_panel: pvt.PVT,
    resolution: Optional[int],
    segment: Segment,
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
        "Beginning calculation of absorber equation for segment %s.",
        segment.coordinates,
    )

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    # import pdb

    # pdb.set_trace(header=f"T_A{segment.coordinates}")

    if operating_mode.dynamic:
        collector_internal_energy_change: float = (
            segment.width  # [m]
            * segment.length  # [m]
            * pvt_panel.absorber.thickness  # [m]
            * pvt_panel.absorber.density  # [kg/m^3]
            * pvt_panel.absorber.heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )
    else:
        collector_internal_energy_change = 0
    logger.debug(
        "Absorber internal energy term: %s W/K", collector_internal_energy_change
    )

    # Compute the positive conductive term based on the next segment along.
    positive_x_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index + 1, segment.y_index)
    )
    if positive_x_segment is not None:
        positive_x_wise_conduction: float = (
            pvt_panel.absorber.conductivity  # [W/m*K]
            * pvt_panel.absorber.thickness  # [m]
            * segment.length  # [m]
            / (0.5 * (segment.width + positive_x_segment.width))  # [m]
        )
    else:
        positive_x_wise_conduction = 0
    logger.debug(
        "Positive absorber x-wise conduction term: %s W/K", positive_x_wise_conduction
    )

    # Compute the positive conductive term based on the next segment along.
    negative_x_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index - 1, segment.y_index)
    )
    if negative_x_segment is not None:
        negative_x_wise_conduction: float = (
            pvt_panel.absorber.conductivity  # [W/m*K]
            * pvt_panel.absorber.thickness  # [m]
            * segment.length  # [m]
            / (0.5 * (segment.width + negative_x_segment.width))  # [m]
        )
    else:
        negative_x_wise_conduction = 0
    logger.debug(
        "Negative absorber x-wise conduction term: %s W/K", negative_x_wise_conduction
    )

    # Compute the overall x-wise conduction term.
    x_wise_conduction = positive_x_wise_conduction + negative_x_wise_conduction
    logger.debug("Absorber x-wise conduction term: %s W/K", x_wise_conduction)

    # Compute the positive conductive term based on the next segment along.
    positive_y_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index, segment.y_index + 1)
    )
    if positive_y_segment is not None:
        positive_y_wise_conduction: float = (
            pvt_panel.absorber.conductivity  # [W/m*K]
            * pvt_panel.absorber.thickness  # [m]
            * segment.width  # [m]
            / (0.5 * (segment.length + positive_y_segment.length))  # [m]
        )
    else:
        positive_y_wise_conduction = 0
    logger.debug(
        "Positive absorber y-wise conduction term: %s W/K", positive_y_wise_conduction
    )

    # Compute the positive conductive term based on the next segment along.
    negative_y_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index, segment.y_index - 1)
    )
    if negative_y_segment is not None:
        negative_y_wise_conduction: float = (
            pvt_panel.absorber.conductivity  # [W/m*K]
            * pvt_panel.absorber.thickness  # [m]
            * segment.width  # [m]
            / (0.5 * (segment.length + negative_y_segment.length))  # [m]
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
            segment.width  # [m]
            * segment.length  # [m]
            / insulation_thermal_resistance(
                best_guess_temperature_vector,
                pvt_panel,
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.absorber,
                    segment.x_index,
                    segment.y_index,
                ),
                weather_conditions,
            )  # [m^2*K/W]
        )
        if not segment.pipe
        else 0
    )
    logger.debug(
        "Absorber to insulation loss term: %s W/K", absorber_to_insulation_loss
    )

    # If there are no upper layers, compute the upward loss terms.
    if not segment.pv and not segment.glass:
        (
            absorber_to_air_conduction,
            absorber_to_sky_radiation,
        ) = physics_utils.upward_loss_terms(
            best_guess_temperature_vector,
            pvt_panel,
            segment,
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index,
                segment.y_index,
            ),
            weather_conditions,
        )
        logger.debug("Absorber to air conduction %s W/K", absorber_to_air_conduction)
        logger.debug("Absorber to sky radiation %s W/K", absorber_to_sky_radiation)
    else:
        absorber_to_air_conduction = 0
        absorber_to_air_radiation = 0

    # Compute the T_A(i, j) term
    row_equation[
        index_handler.index_from_segment_coordinates(
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.absorber,
            segment.x_index,
            segment.y_index,
        )
    ] = (
        collector_internal_energy_change
        + x_wise_conduction
        + y_wise_conduction
        + (pv_to_absorber_conduction if segment.pv else 0)
        + absorber_to_pipe_conduction
        + absorber_to_insulation_loss
        + (
            absorber_to_air_conduction + absorber_to_air_radiation
            if (not segment.pv and not segment.glass)
            else 0
        )
    )

    # Compute the T_A(i+1, j) term provided that that segment exists.
    if segment.x_index + 1 < number_of_x_segments:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.absorber,
                segment.x_index + 1,
                segment.y_index,
            )
        ] = (
            -1 * positive_x_wise_conduction
        )

    # Compute the T_A(i-1, j) term provided that that segment exists.
    if segment.x_index > 0:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.absorber,
                segment.x_index - 1,
                segment.y_index,
            )
        ] = (
            -1 * negative_x_wise_conduction
        )

    # Compute the T_A(i, j+1) term provided that that segment exists.
    if segment.y_index + 1 < number_of_y_segments:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.absorber,
                segment.x_index,
                segment.y_index + 1,
            )
        ] = (
            -1 * positive_y_wise_conduction
        )

    # Compute the T_A(i, j-1) term provided that that segment exists.
    if segment.y_index > 0:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.absorber,
                segment.x_index,
                segment.y_index - 1,
            )
        ] = (
            -1 * negative_y_wise_conduction
        )

    # Compute the T_pv(i, j) term provided that there is a absorber layer present.
    if segment.pv:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index,
                segment.y_index,
            )
        ] = (
            -1 * pv_to_absorber_conduction
        )

    # Compute the T_P(pipe_number, j) term provided that there is a pipe present.
    if segment.pipe:
        if segment.pipe_index is None:
            raise ProgrammerJudgementFault(
                "A segment specifies a pipe is present yet no pipe index is supplied. "
                f"Segment: {str(segment)}"
            )
        try:
            row_equation[
                index_handler.index_from_pipe_coordinates(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.pipe,
                    segment.pipe_index,
                    segment.y_index,
                )
            ] = (
                -1 * absorber_to_pipe_conduction
            )
        except ProgrammerJudgementFault as e:
            raise ProgrammerJudgementFault(
                "Error determining pipe temperature term. "
                f"Likely that pipe index was not specified: {str(e)}"
            ) from None

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Ambient temperature term.
        absorber_to_insulation_loss  # [W/K]
        * weather_conditions.ambient_temperature  # [K]
    )

    if operating_mode.dynamic:
        resultant_vector_value += (
            # Internal heat change term.
            collector_internal_energy_change  # [W/K]
            * previous_temperature_vector[  # type: ignore
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.absorber,
                    segment.x_index,
                    segment.y_index,
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


def _boundary_condition_equations(
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
) -> List[Tuple[List[float], float]]:
    """
    Returns matrix rows and resultant vector values representing boundary conditions.

    These inluce:
        - there is no x temperature gradient at the "left" and "right" edges of the
          panel: physically, this means that the temperatures of cells near the x = 0
          and x = W boundaries are equal. These are represented as equations 1 (at the
          x=0 boundary) and 2 (at the x=W boundary);
        - there is no y temperature gradient along the "bottom" or "top" edges of the
          panel: physically, this means taht the temperatures of cells near the y = 0
          and y = H boundaries are equal. These are represented as equations 3 (at the
          y=0 boundary) and 4 (at the y=H boundary).

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `list` of `tuple`s containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    equations: List[Tuple[List[float], float]] = list()

    # Work through each layer, applying the boundary conditions.
    for temperature_name in {
        TemperatureName.glass,
        TemperatureName.pv,
        TemperatureName.absorber,
    }:
        # Work along both "left" and "right" edges, applying the boundary conditions.
        for y_coord in range(number_of_y_segments):
            # Equation 1: Zero temperature gradient at x=0.
            row_equation: List[float] = [0] * number_of_temperatures
            row_equation[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    0,
                    y_coord,
                )
            ] = -1
            row_equation[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    1,
                    y_coord,
                )
            ] = 1
            equations.append((row_equation, 0))
            # Equation 2: Zero temperature gradient at x=W.
            row_equation = [0] * number_of_temperatures
            row_equation[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    number_of_x_segments - 1,
                    y_coord,
                )
            ] = -1
            row_equation[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    number_of_x_segments - 2,
                    y_coord,
                )
            ] = 1
            equations.append((row_equation, 0))

        # Work along both "top" and "bottom" edges, applying the boundary conditions.
        for x_coord in range(number_of_x_segments):
            # Equation 3: Zero temperature gradient at y=0.
            row_equation = [0] * number_of_temperatures
            row_equation[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    x_coord,
                    0,
                )
            ] = -1
            row_equation[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    x_coord,
                    1,
                )
            ] = 1
            equations.append((row_equation, 0))
            # Equation 4: Zero temperature gradient at y=H.
            row_equation = [0] * number_of_temperatures
            row_equation[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    x_coord,
                    number_of_y_segments - 1,
                )
            ] = -1
            row_equation[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    x_coord,
                    number_of_y_segments - 2,
                )
            ] = 1
            equations.append((row_equation, 0))

    return equations


def _decoupled_system_continuity_equation(
    collector_input_temperature: float,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
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
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
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
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.htf_in,
                pipe_number,
                0,
            )
        ] = 1
        row_equation[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
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
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.htf_out,
                pipe_number,
                number_of_y_segments - 1,
            )
        ] = (
            -1 / number_of_pipes
        )
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector_out,
        )
    ] = 1
    equations.append((row_equation, 0))

    return equations


def _fluid_continuity_equation(
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    segment: Segment,
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
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf_out,
            segment.pipe_index,  # type: ignore
            segment.y_index,
        )
    ] = -1

    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf_in,
            segment.pipe_index,  # type: ignore
            segment.y_index + 1,
        )
    ] = 1

    return row_equation, 0


def _glass_equation(  # pylint: disable=too-many-branches
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    glass_to_pv_conduction: float,
    glass_to_pv_radiation: float,
    logger: logging.Logger,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    previous_temperature_vector: Optional[numpy.ndarray],
    pvt_panel: pvt.PVT,
    resolution: Optional[int],
    segment: Segment,
    weather_conditions: WeatherConditions,
) -> Tuple[List[float], float]:
    """
    Returns a matrix row and resultant vector value representing the glass equation.

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
        "Beginning calculation of glass equation for segment %s.", segment.coordinates
    )

    if operating_mode.dynamic:
        glass_internal_energy: float = (
            segment.width  # [m]
            * segment.length  # [m]
            * pvt_panel.glass.thickness  # [m]
            * pvt_panel.glass.density  # [kg/m^3]
            * pvt_panel.glass.heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )
    else:
        glass_internal_energy = 0
    logger.debug("Glass internal energy term: %s W/K", glass_internal_energy)

    # Compute the positive conductive term based on the next segment along.
    positive_x_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index + 1, segment.y_index)
    )
    if positive_x_segment is not None:
        positive_x_wise_conduction: float = (
            pvt_panel.glass.conductivity  # [W/m*K]
            * pvt_panel.glass.thickness  # [m]
            * segment.length  # [m]
            / (0.5 * (segment.width + positive_x_segment.width))  # [m]
        )
    else:
        positive_x_wise_conduction = 0
    logger.debug(
        "Positive glass x-wise conduction term: %s W/K", positive_x_wise_conduction
    )

    # Compute the positive conductive term based on the next segment along.
    negative_x_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index - 1, segment.y_index)
    )
    if negative_x_segment is not None:
        negative_x_wise_conduction: float = (
            pvt_panel.glass.conductivity  # [W/m*K]
            * pvt_panel.glass.thickness  # [m]
            * segment.length  # [m]
            / (0.5 * (segment.width + negative_x_segment.width))  # [m]
        )
    else:
        negative_x_wise_conduction = 0
    logger.debug(
        "Negative glass x-wise conduction term: %s W/K", negative_x_wise_conduction
    )

    # Compute the overall x-wise conduction term.
    x_wise_conduction = positive_x_wise_conduction + negative_x_wise_conduction
    logger.debug("Glass x-wise conduction term: %s W/K", x_wise_conduction)

    # Compute the positive conductive term based on the next segment along.
    positive_y_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index, segment.y_index + 1)
    )
    if positive_y_segment is not None:
        positive_y_wise_conduction: float = (
            pvt_panel.glass.conductivity  # [W/m*K]
            * pvt_panel.glass.thickness  # [m]
            * segment.width  # [m]
            / (0.5 * (segment.length + positive_y_segment.length))  # [m]
        )
    else:
        positive_y_wise_conduction = 0
    logger.debug(
        "Positive glass y-wise conduction term: %s W/K", positive_y_wise_conduction
    )

    # Compute the positive conductive term based on the next segment along.
    negative_y_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index, segment.y_index - 1)
    )
    if negative_y_segment is not None:
        negative_y_wise_conduction: float = (
            pvt_panel.glass.conductivity  # [W/m*K]
            * pvt_panel.glass.thickness  # [m]
            * segment.width  # [m]
            / (0.5 * (segment.length + negative_y_segment.length))  # [m]
        )
    else:
        negative_y_wise_conduction = 0
    logger.debug(
        "Negative glass y-wise conduction term: %s W/K", negative_y_wise_conduction
    )

    # Compute the overall y-wise conduction term.
    y_wise_conduction = positive_y_wise_conduction + negative_y_wise_conduction
    logger.debug("Glass y-wise conduction term: %s W/K", y_wise_conduction)

    glass_to_air_conduction, glass_to_sky_radiation = physics_utils.upward_loss_terms(
        best_guess_temperature_vector,
        pvt_panel,
        segment,
        index_handler.index_from_segment_coordinates(
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.glass,
            segment.x_index,
            segment.y_index,
        ),
        weather_conditions,
    )
    logger.debug("Glass to air conduction %s W/K", glass_to_air_conduction)
    logger.debug("Glass to sky radiation %s W/K", glass_to_sky_radiation)

    # Compute the T_g(i, j) term
    row_equation[
        index_handler.index_from_segment_coordinates(
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.glass,
            segment.x_index,
            segment.y_index,
        )
    ] = (
        glass_internal_energy
        + x_wise_conduction
        + y_wise_conduction
        + glass_to_air_conduction
        + glass_to_sky_radiation
        + glass_to_pv_conduction
        + glass_to_pv_radiation
    )

    # Compute the T_g(i+1, j) term provided that that segment exists.
    if segment.x_index + 1 < number_of_x_segments:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index + 1,
                segment.y_index,
            )
        ] = (
            -1 * positive_x_wise_conduction
        )

    # Compute the T_g(i-1, j) term provided that that segment exists.
    if segment.x_index > 0:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index - 1,
                segment.y_index,
            )
        ] = (
            -1 * negative_x_wise_conduction
        )

    # Compute the T_g(i, j+1) term provided that that segment exists.
    if segment.y_index + 1 < number_of_y_segments:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index,
                segment.y_index + 1,
            )
        ] = (
            -1 * positive_y_wise_conduction
        )

    # Compute the T_g(i, j-1) term provided that that segment exists.
    if segment.y_index > 0:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index,
                segment.y_index - 1,
            )
        ] = (
            -1 * negative_y_wise_conduction
        )

    # Compute the T_pv(i, j) term provided that there is a PV layer present.
    if segment.pv:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index,
                segment.y_index,
            )
        ] = -1 * (glass_to_pv_conduction + glass_to_pv_radiation)

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Ambient temperature term.
        glass_to_air_conduction * weather_conditions.ambient_temperature  # [W]
        # Sky temperature term.
        + glass_to_sky_radiation * weather_conditions.sky_temperature  # [W]
        # Solar absorption term.
        + segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.glass_transmissivity_absorptivity_product
        * weather_conditions.irradiance  # [W/m^2]
    )

    # if len(numpy.argwhere(numpy.isnan(row_equation))) > 0:
    #     import pdb

    #     pdb.set_trace()

    if operating_mode.dynamic:
        resultant_vector_value += (
            # Previous glass temperature term.
            glass_internal_energy  # [W/K]
            * previous_temperature_vector[  # type: ignore
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.glass,
                    segment.x_index,
                    segment.y_index,
                )
            ]  # [K]
        )

        logger.debug(
            "Rough Glass Temperature estimate: %s K.",
            int(
                resultant_vector_value
                / (
                    glass_to_air_conduction
                    + glass_to_sky_radiation
                    + glass_internal_energy
                )
            ),
        )

    return row_equation, resultant_vector_value


def _htf_continuity_equation(
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    segment: Segment,
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
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf,
            segment.pipe_index,  # type: ignore
            segment.y_index,
        )
    ] = 1

    # Compute the T_f,in(#, 0) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf_in,
            segment.pipe_index,  # type: ignore
            segment.y_index,
        )
    ] = -0.5

    # Compute the T_f,out(#, 0) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf_out,
            segment.pipe_index,  # type: ignore
            segment.y_index,
        )
    ] = -0.5

    return row_equation, 0


def _htf_equation(
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    pipe_to_bulk_water_heat_transfer: float,
    previous_temperature_vector: Optional[numpy.ndarray],
    pvt_panel: pvt.PVT,
    resolution: Optional[int],
    segment: Segment,
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
            * (pvt_panel.absorber.inner_pipe_diameter / 2) ** 2  # [m^2]
            * segment.length  # [m]
            * DENSITY_OF_WATER  # [kg/m^3]
            * pvt_panel.absorber.htf_heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )  # [W/K]
    else:
        bulk_water_internal_energy = 0

    fluid_input_output_transfer_term = (
        pvt_panel.absorber.mass_flow_rate  # [kg/s]
        * pvt_panel.absorber.htf_heat_capacity  # [J/kg*K]
        / pvt_panel.absorber.number_of_pipes
    )  # [W/K]

    # Compute the T_f(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf,
            segment.pipe_index,  # type: ignore
            segment.y_index,
        )
    ] = (
        bulk_water_internal_energy + pipe_to_bulk_water_heat_transfer
    )

    # Compute the T_f,in(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf_in,
            segment.pipe_index,  # type: ignore
            segment.y_index,
        )
    ] = (
        -1 * fluid_input_output_transfer_term
    )

    # Compute the T_f,out(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf_out,
            segment.pipe_index,  # type: ignore
            segment.y_index,
        )
    ] = fluid_input_output_transfer_term

    # Compute the T_P(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.pipe,
            segment.pipe_index,  # type: ignore
            segment.y_index,
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
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.htf,
                    segment.pipe_index,  # type: ignore
                    segment.y_index,
                )
            ]
        )

    return row_equation, resultant_vector_value


def _pipe_equation(
    absorber_to_pipe_conduction: float,
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    pipe_to_htf_heat_transfer: float,
    previous_temperature_vector: Optional[numpy.ndarray],
    pvt_panel: pvt.PVT,
    resolution: Optional[int],
    segment: Segment,
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
                (pvt_panel.absorber.outer_pipe_diameter / 2) ** 2  # [m^2]
                - (pvt_panel.absorber.inner_pipe_diameter / 2) ** 2  # [m^2]
            )
            * segment.length  # [m]
            * pvt_panel.absorber.pipe_density  # [kg/m^3]
            * pvt_panel.absorber.heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )
    else:
        pipe_internal_heat_change = 0

    pipe_to_surroundings_losses = (
        numpy.pi
        * (pvt_panel.absorber.outer_pipe_diameter / 2)  # [m]
        * segment.length  # [m]
        / insulation_thermal_resistance(
            best_guess_temperature_vector,
            pvt_panel,
            index_handler.index_from_pipe_coordinates(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pipe,
                segment.pipe_index,  # type: ignore
                segment.y_index,
            ),
            weather_conditions,
        )  # [K*m^2/W]
    )

    # Compute the T_P(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.pipe,
            segment.pipe_index,  # type: ignore
            segment.y_index,
        )
    ] = (
        pipe_internal_heat_change  # [W/K]
        + absorber_to_pipe_conduction  # [W/K]
        + pipe_to_htf_heat_transfer  # [W/K]
        + pipe_to_surroundings_losses
    )

    # Compute the T_A(i, j) term.
    row_equation[
        index_handler.index_from_segment_coordinates(
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.absorber,
            segment.x_index,
            segment.y_index,
        )
    ] = -1 * (absorber_to_pipe_conduction)

    # Compute the T_f(#, j) term.
    row_equation[
        index_handler.index_from_pipe_coordinates(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf,
            segment.pipe_index,  # type: ignore
            segment.y_index,
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
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.pipe,
                    segment.pipe_index,  # type: ignore
                    segment.y_index,
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


def _pv_equation(  # pylint: disable=too-many-branches
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    logger: logging.Logger,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    previous_temperature_vector: Optional[numpy.ndarray],
    pv_to_absorber_conduction: float,
    pv_to_glass_conduction: float,
    pv_to_glass_radiation: float,
    pvt_panel: pvt.PVT,
    resolution: Optional[int],
    segment: Segment,
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

    #     pdb.set_trace(header=f"T_PV{segment.coordinates}")

    logger.debug(
        "Beginning calculation of PV equation for segment %s.", segment.coordinates
    )

    if operating_mode.dynamic:
        pv_internal_energy: float = (
            segment.width  # [m]
            * segment.length  # [m]
            * pvt_panel.pv.thickness  # [m]
            * pvt_panel.pv.density  # [kg/m^3]
            * pvt_panel.pv.heat_capacity  # [J/kg*K]
            / resolution  # type: ignore  # [s]
        )
    else:
        pv_internal_energy = 0
    logger.debug("PV internal energy term: %s W/K", pv_internal_energy)

    # Compute the positive conductive term based on the next segment along.
    positive_x_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index + 1, segment.y_index)
    )
    if positive_x_segment is not None:
        positive_x_wise_conduction: float = (
            pvt_panel.pv.conductivity  # [W/m*K]
            * pvt_panel.pv.thickness  # [m]
            * segment.length  # [m]
            / (0.5 * (segment.width + positive_x_segment.width))  # [m]
        )
    else:
        positive_x_wise_conduction = 0
    logger.debug(
        "Positive PV x-wise conduction term: %s W/K", positive_x_wise_conduction
    )

    # Compute the positive conductive term based on the next segment along.
    negative_x_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index - 1, segment.y_index)
    )
    if negative_x_segment is not None:
        negative_x_wise_conduction: float = (
            pvt_panel.pv.conductivity  # [W/m*K]
            * pvt_panel.pv.thickness  # [m]
            * segment.length  # [m]
            / (0.5 * (segment.width + negative_x_segment.width))  # [m]
        )
    else:
        negative_x_wise_conduction = 0
    logger.debug(
        "Negative PV x-wise conduction term: %s W/K", negative_x_wise_conduction
    )

    # Compute the overall x-wise conduction term.
    x_wise_conduction = positive_x_wise_conduction + negative_x_wise_conduction
    logger.debug("PV x-wise conduction term: %s W/K", x_wise_conduction)

    # Compute the positive conductive term based on the next segment along.
    positive_y_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index, segment.y_index + 1)
    )
    if positive_y_segment is not None:
        positive_y_wise_conduction: float = (
            pvt_panel.pv.conductivity  # [W/m*K]
            * pvt_panel.pv.thickness  # [m]
            * segment.width  # [m]
            / (0.5 * (segment.length + positive_y_segment.length))  # [m]
        )
    else:
        positive_y_wise_conduction = 0
    logger.debug(
        "Positive PV y-wise conduction term: %s W/K", positive_y_wise_conduction
    )

    # Compute the positive conductive term based on the next segment along.
    negative_y_segment = pvt_panel.segments.get(
        SegmentCoordinates(segment.x_index, segment.y_index - 1)
    )
    if negative_y_segment is not None:
        negative_y_wise_conduction: float = (
            pvt_panel.pv.conductivity  # [W/m*K]
            * pvt_panel.pv.thickness  # [m]
            * segment.width  # [m]
            / (0.5 * (segment.length + negative_y_segment.length))  # [m]
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
        index_handler.index_from_segment_coordinates(
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.pv,
            segment.x_index,
            segment.y_index,
        )
    ] = (
        pv_internal_energy
        + x_wise_conduction
        + y_wise_conduction
        + pv_to_glass_radiation
        + pv_to_glass_conduction
        + pv_to_absorber_conduction
    )

    # Compute the T_pv(i+1, j) term provided that that segment exists.
    if segment.x_index + 1 < number_of_x_segments:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index + 1,
                segment.y_index,
            )
        ] = (
            -1 * positive_x_wise_conduction
        )

    # Compute the T_pv(i-1, j) term provided that that segment exists.
    if segment.x_index > 0:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index - 1,
                segment.y_index,
            )
        ] = (
            -1 * negative_x_wise_conduction
        )

    # Compute the T_pv(i, j+1) term provided that that segment exists.
    if segment.y_index + 1 < number_of_y_segments:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index,
                segment.y_index + 1,
            )
        ] = (
            -1 * positive_y_wise_conduction
        )

    # Compute the T_pv(i, j-1) term provided that that segment exists.
    if segment.y_index > 0:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index,
                segment.y_index - 1,
            )
        ] = (
            -1 * negative_y_wise_conduction
        )

    # Compute the T_g(i, j) term provided that there is a glass layer present.
    if segment.glass:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index,
                segment.y_index,
            )
        ] = -1 * (pv_to_glass_conduction + pv_to_glass_radiation)

    # Compute the T_A(i, j) term provided that there is a absorber layer present.
    if segment.absorber:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.absorber,
                segment.x_index,
                segment.y_index,
            )
        ] = -1 * (pv_to_absorber_conduction)

    solar_thermal_resultant_vector_absorbtion_term = (
        pvt_panel.pv_transmissivity_absorptivity_product
        * weather_conditions.irradiance  # [W/m^2]
        * segment.width  # [m]
        * segment.length  # [m]
    ) - (
        weather_conditions.irradiance  # [W/m^2]
        * segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.pv.reference_efficiency
        * (
            1
            - pvt_panel.pv.thermal_coefficient
            * (
                best_guess_temperature_vector[
                    index_handler.index_from_segment_coordinates(
                        number_of_x_segments,
                        number_of_y_segments,
                        TemperatureName.pv,
                        segment.x_index,
                        segment.y_index,
                    )
                ]
                - pvt_panel.pv.reference_temperature
            )
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
    )

    if operating_mode.dynamic:
        resultant_vector_value += (
            # Internal energy change
            pv_internal_energy  # [W/K]
            * previous_temperature_vector[  # type: ignore
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.pv,
                    segment.x_index,
                    segment.y_index,
                )
            ]
        )

    return row_equation, resultant_vector_value


def _system_continuity_equations(
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: Union[List[float], numpy.ndarray],
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
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.htf_in,
                pipe_number,
                0,
            )
        ] = 1
        resultant_value = previous_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
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
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.htf_out,
                pipe_number,
                number_of_y_segments - 1,
            )
        ] = (
            -1 / number_of_pipes
        )
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector_out,
        )
    ] = 1
    equations.append((row_equation, 0))

    # Equation 3: Fluid leaving the absorber enters the tank without losses.
    row_equation = [0] * number_of_temperatures
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector_out,
        )
    ] = -1
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.tank_in,
        )
    ] = 1
    equations.append((row_equation, 0))

    row_equation = [0] * number_of_temperatures
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.tank_out,
        )
    ] = -1
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector_in,
        )
    ] = 1
    equations.append((row_equation, 0))

    return equations


def _tank_continuity_equation(
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    heat_exchanger: exchanger.Exchanger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
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
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank_in,
            )
        ]
        > best_guess_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank,
            )
        ]
    ):
        row_equation[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank,
            )
        ] = (
            -1 * heat_exchanger.efficiency
        )
        row_equation[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank_in,
            )
        ] = (
            heat_exchanger.efficiency - 1
        )
        row_equation[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank_out,
            )
        ] = 1
        return row_equation, 0

    # Otherwise, the flow is diverted back into the absorber.
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.tank_in,
        )
    ] = -1
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.tank_out,
        )
    ] = 1
    return row_equation, 0


def _tank_equation(
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    heat_exchanger: exchanger.Exchanger,
    hot_water_load: float,
    hot_water_tank: tank.Tank,
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: Optional[numpy.ndarray],
    pvt_panel: pvt.PVT,
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
            pvt_panel.absorber.mass_flow_rate  # [kg/s]
            * pvt_panel.absorber.htf_heat_capacity  # [J/kg*K]
            * heat_exchanger.efficiency
        )
        if best_guess_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank_in,
            )
        ]
        > best_guess_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank,
            )
        ]
        else 0
    )
    logger.debug("Tank heat-input term: %s W/K", heat_input_term)

    # Compute the T_t term
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.tank,
        )
    ] = (
        tank_internal_energy + hot_water_load_term + heat_loss_term + heat_input_term
    )

    # Compute the T_c,out term
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
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
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
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


##################
# Public methods #
##################


def calculate_matrix_equation(
    *,
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    pvt_panel: pvt.PVT,
    resolution: Optional[int],
    weather_conditions: WeatherConditions,
    collector_input_temperature: Optional[float] = None,
    heat_exchanger: Optional[exchanger.Exchanger] = None,
    hot_water_load: Optional[float] = None,
    hot_water_tank: Optional[tank.Tank] = None,
    previous_temperature_vector: Optional[numpy.ndarray] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Calculates and returns both the matrix and resultant vector for the matrix equation.

    :return:
        A `tuple` containing both the matrix "A" and resultant vector "B" for the matrix
        equation representing the temperature equations.

    """

    logger.info("Matrix module called: calculating matrix and resultant vector.")
    logger.info(
        "A %s and %s matrix will be computed.",
        "dynamic" if operating_mode.dynamic else "steady-state",
        "decoupled" if operating_mode.decoupled else "coupled",
    )

    # Instantiate an empty matrix and array based on the number of temperatures present.
    matrix = numpy.zeros([0, number_of_temperatures])
    resultant_vector = numpy.zeros([0, 1])

    for segment_coordinates, segment in pvt_panel.segments.items():
        logger.debug("Calculating equations for segment %s", segment_coordinates)
        # Compute the various shared values.
        glass_to_pv_conduction = (
            segment.width
            * segment.length
            / air_gap_resistance(
                pvt_panel,
                0.5
                * (
                    best_guess_temperature_vector[
                        index_handler.index_from_segment_coordinates(
                            number_of_x_segments,
                            number_of_y_segments,
                            TemperatureName.pv,
                            segment.x_index,
                            segment.y_index,
                        )
                    ]
                    + best_guess_temperature_vector[
                        index_handler.index_from_segment_coordinates(
                            number_of_x_segments,
                            number_of_y_segments,
                            TemperatureName.glass,
                            segment.x_index,
                            segment.y_index,
                        )
                    ]
                ),
                weather_conditions,
            )
        )
        logger.debug("Glass to pv conduction %s W/K", glass_to_pv_conduction)

        glass_to_pv_radiation = (
            segment.width
            * segment.length
            * radiative_heat_transfer_coefficient(
                destination_emissivity=pvt_panel.pv.emissivity,
                destination_temperature=best_guess_temperature_vector[
                    index_handler.index_from_segment_coordinates(
                        number_of_x_segments,
                        number_of_y_segments,
                        TemperatureName.pv,
                        segment.x_index,
                        segment.y_index,
                    )
                ],
                source_emissivity=pvt_panel.glass.emissivity,
                source_temperature=best_guess_temperature_vector[
                    index_handler.index_from_segment_coordinates(
                        number_of_x_segments,
                        number_of_y_segments,
                        TemperatureName.glass,
                        segment.x_index,
                        segment.y_index,
                    )
                ],
            )
        )
        logger.debug("Glass to pv radiation %s W/K", glass_to_pv_radiation)

        pv_to_absorber_conduction = (
            segment.width * segment.length / pvt_panel.pv_to_absorber_thermal_resistance
        )
        logger.debug("PV to absorber conduction: %s W/K", pv_to_absorber_conduction)

        absorber_to_pipe_conduction = (
            (
                segment.width  # [m]
                * segment.length  # [m]
                * pvt_panel.bond.conductivity  # [W/m*K]
                / pvt_panel.bond.thickness  # [m]
            )
            if segment.pipe
            else 0
        )
        logger.debug("Absorber to pipe conduction: %s W/K", absorber_to_pipe_conduction)

        pipe_to_htf_heat_transfer = (
            segment.length  # [m]
            * numpy.pi
            * pvt_panel.absorber.inner_pipe_diameter  # [m]
            * pvt_panel.absorber.convective_heat_transfer_coefficient_of_water  # [W/m^2*K]
        )
        logger.debug("Pipe to HTF heat transfer: %s W/K", pipe_to_htf_heat_transfer)

        if segment.glass:
            glass_equation, glass_resultant_value = _glass_equation(
                best_guess_temperature_vector,
                glass_to_pv_conduction,
                glass_to_pv_radiation,
                logger,
                number_of_temperatures,
                number_of_x_segments,
                number_of_y_segments,
                operating_mode,
                previous_temperature_vector,
                pvt_panel,
                resolution,
                segment,
                weather_conditions,
            )
            logger.debug(
                "Glass equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
                segment.coordinates,
                ", ".join([f"{value:.3f} W/K" for value in glass_equation]),
                glass_resultant_value,
            )
            matrix = numpy.vstack((matrix, glass_equation))
            resultant_vector = numpy.vstack((resultant_vector, glass_resultant_value))

        if segment.pv:
            pv_equation, pv_resultant_value = _pv_equation(
                best_guess_temperature_vector,
                logger,
                number_of_temperatures,
                number_of_x_segments,
                number_of_y_segments,
                operating_mode,
                previous_temperature_vector,
                pv_to_absorber_conduction,
                glass_to_pv_conduction,
                glass_to_pv_radiation,
                pvt_panel,
                resolution,
                segment,
                weather_conditions,
            )
            logger.debug(
                "PV equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
                segment.coordinates,
                ", ".join([f"{value:.3f} W/K" for value in pv_equation]),
                pv_resultant_value,
            )
            matrix = numpy.vstack((matrix, pv_equation))
            resultant_vector = numpy.vstack((resultant_vector, pv_resultant_value))

        if segment.absorber:
            absorber_equation, absorber_resultant_value = _absorber_equation(
                absorber_to_pipe_conduction,
                best_guess_temperature_vector,
                logger,
                number_of_pipes,
                number_of_temperatures,
                number_of_x_segments,
                number_of_y_segments,
                operating_mode,
                previous_temperature_vector,
                pv_to_absorber_conduction,
                pvt_panel,
                resolution,
                segment,
                weather_conditions,
            )
            logger.debug(
                "Collector equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
                segment.coordinates,
                ", ".join([f"{value:.3f} W/K" for value in absorber_equation]),
                absorber_resultant_value,
            )
            matrix = numpy.vstack((matrix, absorber_equation))
            resultant_vector = numpy.vstack(
                (resultant_vector, absorber_resultant_value)
            )

        # Only calculate the pipe equations if the segment has an associated pipe.
        if not segment.pipe:
            logger.debug("3 equations for segment %s", segment_coordinates)
            continue

        pipe_equation, pipe_resultant_value = _pipe_equation(
            absorber_to_pipe_conduction,
            best_guess_temperature_vector,
            logger,
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            operating_mode,
            pipe_to_htf_heat_transfer,
            previous_temperature_vector,
            pvt_panel,
            resolution,
            segment,
            weather_conditions,
        )
        logger.debug(
            "Pipe equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in pipe_equation]),
            pipe_resultant_value,
        )
        matrix = numpy.vstack((matrix, pipe_equation))
        resultant_vector = numpy.vstack((resultant_vector, pipe_resultant_value))

        htf_equation, htf_resultant_value = _htf_equation(
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            operating_mode,
            pipe_to_htf_heat_transfer,
            previous_temperature_vector,
            pvt_panel,
            resolution,
            segment,
        )
        logger.debug(
            "HTF equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in htf_equation]),
            htf_resultant_value,
        )
        matrix = numpy.vstack((matrix, htf_equation))
        resultant_vector = numpy.vstack((resultant_vector, htf_resultant_value))

        htf_equation, htf_resultant_value = _htf_continuity_equation(
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            segment,
        )
        logger.debug(
            "HTF definition equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in htf_equation]),
            htf_resultant_value,
        )
        matrix = numpy.vstack((matrix, htf_equation))
        resultant_vector = numpy.vstack((resultant_vector, htf_resultant_value))

        # Fluid continuity equations only need to be computed if there exist multiple
        # connected segments.
        if segment.y_index >= number_of_y_segments - 1:
            logger.debug("6 equations for segment %s", segment_coordinates)
            continue

        (
            fluid_continuity_equation,
            fluid_continuity_resultant_value,
        ) = _fluid_continuity_equation(
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            segment,
        )
        logger.debug(
            "Fluid continuity equation for segment %s computed:\nEquation: %s\n"
            "Resultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in fluid_continuity_equation]),
            fluid_continuity_resultant_value,
        )
        matrix = numpy.vstack((matrix, fluid_continuity_equation))
        resultant_vector = numpy.vstack(
            (resultant_vector, fluid_continuity_resultant_value)
        )
        logger.debug("7 equations for segment %s", segment_coordinates)

    # # Calculate the system boundary condition equations.
    # boundary_condition_equations = _boundary_condition_equations(
    #     number_of_temperatures, number_of_x_segments, number_of_y_segments
    # )

    # for equation, resultant_value in boundary_condition_equations:
    #     logger.debug(
    #         "Boundary condition equation computed:\nEquation: %s\nResultant value: %s W",
    #         ", ".join([f"{value:.3f} W/K" for value in equation]),
    #         resultant_value,
    #     )
    #     matrix = numpy.vstack((matrix, equation))
    #     resultant_vector = numpy.vstack((resultant_vector, resultant_value))
    #     # if len(matrix) == number_of_temperatures:
    #     #     return matrix, resultant_vector

    # If the system is decoupled, do not compute and add the tank-related equations.
    if operating_mode.decoupled:
        if collector_input_temperature is None:
            raise ProgrammerJudgementFault(
                "{}No collector input temperature passed to the matrix module ".format(
                    BColours.FAIL
                )
                + "for decoupled operation.{}".format(BColours.ENDC)
            )
        decoupled_system_continuity_equations = _decoupled_system_continuity_equation(
            collector_input_temperature,
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
        )
        for equation, resultant_value in decoupled_system_continuity_equations:
            logger.debug(
                "System continuity equation computed:\nEquation: %s\nResultant value: %s W",
                ", ".join([f"{value:.3f} W/K" for value in equation]),
                resultant_value,
            )
            matrix = numpy.vstack((matrix, equation))
            resultant_vector = numpy.vstack((resultant_vector, resultant_value))
        logger.info("Matrix equation computed, matrix dimensions: %s", matrix.shape)

        return matrix, resultant_vector

    if (
        heat_exchanger is None
        or hot_water_load is None
        or hot_water_tank is None
        or previous_temperature_vector is None
    ):
        raise ProgrammerJudgementFault(
            "{}Insufficient parameters for dynamic run:{}{}{}{}{}".format(
                " Heat exchanger missing." if heat_exchanger is None else "",
                " Hot water load missing." if hot_water_load is None else "",
                " Hot-water tank missing." if hot_water_tank is None else "",
                " Previous temperature vector is missing."
                if previous_temperature_vector is None
                else "",
                BColours.FAIL,
                BColours.ENDC,
            )
        )

    # Calculate the tank equations.
    equation, resultant_value = _tank_equation(
        best_guess_temperature_vector,
        heat_exchanger,
        hot_water_load,
        hot_water_tank,
        logger,
        number_of_pipes,
        number_of_temperatures,
        number_of_x_segments,
        number_of_y_segments,
        previous_temperature_vector,
        pvt_panel,
        resolution,
        weather_conditions,
    )
    logger.debug(
        "Tank equation computed:\nEquation: %s\nResultant value: %s W",
        equation,
        resultant_value,
    )
    matrix = numpy.vstack((matrix, equation))
    resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    # Calculate the tank continuity equation.
    equation, resultant_value = _tank_continuity_equation(
        best_guess_temperature_vector,
        heat_exchanger,
        number_of_pipes,
        number_of_temperatures,
        number_of_x_segments,
        number_of_y_segments,
    )
    logger.debug(
        "Tank continuity equation computed:\nEquation: %s\nResultant value: %s W",
        equation,
        resultant_value,
    )
    matrix = numpy.vstack((matrix, equation))
    resultant_vector = numpy.vstack((resultant_vector, resultant_value))
    logger.debug("2 tank equations computed.")

    # Compute the system continuity equations and assign en masse.
    system_continuity_equations = _system_continuity_equations(
        number_of_pipes,
        number_of_temperatures,
        number_of_x_segments,
        number_of_y_segments,
        previous_temperature_vector,
    )
    logger.debug(
        "%s system continuity equations computed.", len(system_continuity_equations)
    )

    for equation, resultant_value in system_continuity_equations:
        logger.debug(
            "System continuity equation computed:\nEquation: %s\nResultant value: %s W",
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    logger.info("Matrix equation computed, matrix dimensions: %s", matrix.shape)

    return matrix, resultant_vector
