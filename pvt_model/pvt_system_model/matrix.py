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

from typing import List, Tuple

import logging
import numpy

from . import exchanger, index_handler, tank
from .pvt_panel import pvt
from .pvt_panel.segment import Segment

from ..__utils__ import TemperatureName, ProgrammerJudgementFault
from .__utils__ import WeatherConditions, PVT_SYSTEM_MODEL_LOGGER_NAME
from .constants import DENSITY_OF_WATER, HEAT_CAPACITY_OF_WATER
from .physics_utils import (
    radiative_heat_transfer_coefficient,
    transmissivity_absorptivity_product,
)

__all__ = ("calculate_matrix_equation",)


####################
# Internal methods #
####################


def _absorber_equation(
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
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

    absorber_internal_energy_change = (
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.collector.thickness  # [m]
        * pvt_panel.collector.density  # [kg/m^3]
        * pvt_panel.collector.heat_capacity  # [J/kg*K]
        / resolution  # [s]
    )
    logger.debug(
        "Absorber internal energy term: %s W/K", absorber_internal_energy_change
    )

    x_wise_conduction = (
        (
            2
            - (1 if segment.x_index == 0 else 0)
            - (1 if segment.x_index == number_of_x_segments - 1 else 0)
        )
        * pvt_panel.collector.conductivity  # [W/m*K]
        * pvt_panel.collector.thickness  # [m]
        * segment.length  # [m]
        / segment.width  # [m]
    )
    logger.debug(
        "Absorber x-wise conduction term: %s W/K",
        x_wise_conduction,
    )

    y_wise_conduction = (
        (
            2
            - (1 if segment.y_index == 0 else 0)
            - (1 if segment.y_index == number_of_y_segments - 1 else 0)
        )
        * pvt_panel.collector.conductivity  # [W/m*K]
        * pvt_panel.collector.thickness  # [m]
        * segment.width  # [m]
        / segment.length  # [m]
    )
    logger.debug("Absorber y-wise conduction term: %s W/K", y_wise_conduction)

    pv_to_absorber_conduction = (
        segment.width * segment.length / pvt_panel.pv_to_collector_thermal_resistance
    )
    logger.debug("PV to absorber conduction term: %s W/K", pv_to_absorber_conduction)

    absorber_to_pipe_conduction = (
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.bond.conductivity  # [W/m*K]
        / pvt_panel.bond.thickness  # [m]
    )
    logger.debug(
        "Absorber to pipe conduction term: %s W/K", absorber_to_pipe_conduction
    )

    absorber_to_insulation_loss = (
        segment.width * segment.length * pvt_panel.insulation_thermal_resistance
    )
    logger.debug(
        "Absorber to insulation loss term: %s W/K", absorber_to_insulation_loss
    )

    # Compute the T_A(i, j) term
    row_equation[
        index_handler.index_from_segment_coordinates(
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector,
            segment.x_index,
            segment.y_index,
        )
    ] = (
        absorber_internal_energy_change
        + x_wise_conduction
        + y_wise_conduction
        + pv_to_absorber_conduction
        + (absorber_to_pipe_conduction if segment.pipe else 0)
        + (absorber_to_insulation_loss if not segment.pipe else 0)
    )

    # Compute the T_A(i+1, j) term provided that that segment exists.
    if segment.x_index + 1 < number_of_x_segments:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index + 1,
                segment.y_index,
            )
        ] = (
            -1 * x_wise_conduction
        )

    # Compute the T_A(i-1, j) term provided that that segment exists.
    if segment.x_index > 0:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index - 1,
                segment.y_index,
            )
        ] = (
            -1 * x_wise_conduction
        )

    # Compute the T_A(i, j+1) term provided that that segment exists.
    if segment.y_index + 1 < number_of_y_segments:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index,
                segment.y_index + 1,
            )
        ] = (
            -1 * y_wise_conduction
        )

    # Compute the T_A(i, j-1) term provided that that segment exists.
    if segment.y_index > 0:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index,
                segment.y_index - 1,
            )
        ] = (
            -1 * y_wise_conduction
        )

    # Compute the T_pv(i, j) term provided that there is a collector layer present.
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
        # Internal heat change term.
        absorber_internal_energy_change  # [W/K]
        * previous_temperature_vector[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index,
                segment.y_index,
            )
        ]  # [K]
        # Ambient temperature term.
        + (
            absorber_to_insulation_loss  # [W/K]
            * weather_conditions.ambient_temperature  # [K]
            if not segment.pipe
            else 0
        )  # [W]
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
        TemperatureName.collector,
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


def _glass_equation(
    best_guess_temperature_vector: numpy.ndarray,
    logger: logging.Logger,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
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

    # if weather_conditions.irradiance > 500:
    #     import pdb

    #     pdb.set_trace(header=f"T_G{segment.coordinates}")

    logger.debug(
        "Beginning calculation of glass equation for segment %s.", segment.coordinates
    )

    glass_internal_energy = (
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.glass.thickness  # [m]
        * pvt_panel.glass.density  # [kg/m^3]
        * pvt_panel.glass.heat_capacity  # [J/kg*K]
        / resolution  # [s]
    )
    logger.debug("Glass internal energy term: %s W/K", glass_internal_energy)

    x_wise_conduction = (
        (
            2
            - (1 if segment.x_index == 0 else 0)
            - (1 if segment.x_index == number_of_x_segments - 1 else 0)
        )
        * pvt_panel.glass.conductivity  # [W/m*K]
        * pvt_panel.glass.thickness  # [m]
        * segment.length  # [m]
        / segment.width  # [m]
    )
    logger.debug("Glass x-wise conduction term: %s W/K", x_wise_conduction)

    y_wise_conduction = (
        (
            2
            - (1 if segment.y_index == 0 else 0)
            - (1 if segment.y_index == number_of_y_segments - 1 else 0)
        )
        * pvt_panel.glass.conductivity  # [W/m*K]
        * pvt_panel.glass.thickness  # [m]
        * segment.width  # [m]
        / segment.length  # [m]
    )
    logger.debug("Glass y-wise conduction term: %s W/K", y_wise_conduction)

    glass_to_air_conduction = (
        +segment.width  # [m]
        * segment.length  # [m]
        * weather_conditions.wind_heat_transfer_coefficient  # [W/m^2*K]
    )
    logger.debug("Glass to air conduction %s W/K", glass_to_air_conduction)

    glass_to_sky_radiation = (
        segment.width  # [m]
        * segment.length  # [m]
        * radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
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
    logger.debug("Glass to sky radiation %s W/K", glass_to_sky_radiation)

    glass_to_pv_conduction = (
        segment.width * segment.length / pvt_panel.air_gap_resistance
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
    logger.debug("Glass to pv radiation %s W/K", glass_to_pv_conduction)

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
            -1 * x_wise_conduction
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
            -1 * x_wise_conduction
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
            -1 * y_wise_conduction
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
            -1 * y_wise_conduction
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
        # Previous glass temperature term.
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.glass.thickness  # [m]
        * pvt_panel.glass.density  # [kg/m^3]
        * pvt_panel.glass.heat_capacity  # [J/kg*K]
        * previous_temperature_vector[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index,
                segment.y_index,
            )
        ]
        / resolution
        # Ambient temperature term.
        + segment.width  # [m]
        * segment.length  # [m]
        * weather_conditions.wind_heat_transfer_coefficient  # [W/m^2*K]
        * weather_conditions.ambient_temperature  # [K]
        # Sky temperature term.
        + segment.width  # [m]
        * segment.length  # [m]
        * radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
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
        * weather_conditions.sky_temperature  # [K]
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
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
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
        # Internal heat change.
        numpy.pi
        * (pvt_panel.collector.inner_pipe_diameter / 2) ** 2  # [m^2]
        * segment.length  # [m]
        * DENSITY_OF_WATER  # [kg/m^3]
        * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
        / resolution  # [s]
        # Heat transfer from the pipe.
        + segment.length  # [m]
        * numpy.pi
        * pvt_panel.collector.inner_pipe_diameter  # [m]
        * pvt_panel.collector.convective_heat_transfer_coefficient_of_water  # [W/m^2*K]
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
    ] = -1 * (
        pvt_panel.collector.mass_flow_rate  # [kg/s]
        * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
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
    ] = (
        pvt_panel.collector.mass_flow_rate  # [kg/s]
        * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
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
    ] = -1 * (
        # Heat transfer from the pipe.
        segment.length  # [m]
        * numpy.pi
        * pvt_panel.collector.inner_pipe_diameter  # [m]
        * pvt_panel.collector.convective_heat_transfer_coefficient_of_water  # [W/m^2*K]
    )

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Internal heat change.
        numpy.pi
        * (pvt_panel.collector.inner_pipe_diameter / 2) ** 2  # [m^2]
        * segment.length  # [m]
        * DENSITY_OF_WATER  # [kg/m^3]
        * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
        * previous_temperature_vector[
            index_handler.index_from_pipe_coordinates(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.htf,
                segment.pipe_index,  # type: ignore
                segment.y_index,
            )
        ]
        / resolution  # [s]
    )

    return row_equation, resultant_vector_value


def _pipe_equation(
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
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

    pipe_internal_heat_change = (
        numpy.pi
        * (
            (pvt_panel.collector.outer_pipe_diameter / 2) ** 2  # [m^2]
            - (pvt_panel.collector.inner_pipe_diameter / 2) ** 2  # [m^2]
        )
        * segment.length  # [m]
        * pvt_panel.collector.density  # [kg/m^3]
        * pvt_panel.collector.heat_capacity  # [J/kg*K]
        / resolution  # [s]
    )

    absorber_to_pipe_conduction = (
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.bond.conductivity  # [W/m*K]
        / pvt_panel.bond.thickness  # [m]
    )

    pipe_to_htf_heat_transfer = (
        segment.length  # [m]
        * numpy.pi
        * pvt_panel.collector.inner_pipe_diameter  # [m]
        * pvt_panel.collector.convective_heat_transfer_coefficient_of_water  # [W/m^2*K]
    )

    pipe_to_surroundings_losses = (
        numpy.pi
        * pvt_panel.collector.outer_pipe_diameter  # [m]
        / pvt_panel.insulation_thermal_resistance  # [K*m/W]
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
            TemperatureName.collector,
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
        # Internal heat change.
        pipe_internal_heat_change
        * previous_temperature_vector[
            index_handler.index_from_pipe_coordinates(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pipe,
                segment.pipe_index,  # type: ignore
                segment.y_index,
            )
        ]
        # Ambient heat loss.
        + pipe_to_surroundings_losses * weather_conditions.ambient_temperature  # [K]
    )

    return row_equation, resultant_vector_value


def _pv_equation(
    best_guess_temperature_vector: numpy.ndarray,
    logger: logging.Logger,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
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

    pv_internal_energy = (
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.pv.thickness  # [m]
        * pvt_panel.pv.density  # [kg/m^3]
        * pvt_panel.pv.heat_capacity  # [J/kg*K]
        / resolution  # [s]
    )
    logger.debug("PV internal energy term: %s W/K", pv_internal_energy)

    x_wise_conduction = (
        (
            2
            - (1 if segment.x_index == 0 else 0)
            - (1 if segment.x_index == number_of_x_segments - 1 else 0)
        )
        * pvt_panel.pv.conductivity  # [W/m*K]
        * pvt_panel.pv.thickness  # [m]
        * segment.length  # [m]
        / segment.width  # [m]
    )
    logger.debug("PV x-wise conduction term: %s W/K", x_wise_conduction)

    y_wise_conduction = (
        (
            2
            - (1 if segment.y_index == 0 else 0)
            - (1 if segment.y_index == number_of_y_segments - 1 else 0)
        )
        * pvt_panel.pv.conductivity  # [W/m*K]
        * pvt_panel.pv.thickness  # [m]
        * segment.width  # [m]
        / segment.length  # [m]
    )
    logger.debug("PV y-wise conduction term: %s W/K", y_wise_conduction)

    pv_to_glass_radiation = (
        segment.width
        * segment.length
        * radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=best_guess_temperature_vector[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.glass,
                    segment.x_index,
                    segment.y_index,
                )
            ],
            source_emissivity=pvt_panel.pv.emissivity,
            source_temperature=best_guess_temperature_vector[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.pv,
                    segment.x_index,
                    segment.y_index,
                )
            ],
        )
    )
    logger.debug("PV to glass radiation term: %s W/K", pv_to_glass_radiation)

    pv_to_glass_conduction = (
        segment.width * segment.length / pvt_panel.air_gap_resistance
    )
    logger.debug("PV to glass conduction term: %s W/K", pv_to_glass_conduction)

    pv_to_absorber_conduction = (
        segment.width * segment.length / pvt_panel.pv_to_collector_thermal_resistance
    )
    logger.debug("PV to absorber conduction term: %s W/K", pv_to_absorber_conduction)

    solar_thermal_absorbtion_term = -1 * (
        transmissivity_absorptivity_product(
            diffuse_reflection_coefficient=pvt_panel.glass.diffuse_reflection_coefficient,
            glass_transmissivity=pvt_panel.glass.transmissivity,
            layer_absorptivity=pvt_panel.pv.absorptivity,
        )
        * weather_conditions.irradiance  # [W/m^2]
        * segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.pv.reference_efficiency
        * pvt_panel.pv.thermal_coefficient
    )

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
        + solar_thermal_absorbtion_term
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
            -1 * x_wise_conduction
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
            -1 * x_wise_conduction
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
            -1 * y_wise_conduction
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
            -1 * y_wise_conduction
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

    # Compute the T_A(i, j) term provided that there is a collector layer present.
    if segment.collector:
        row_equation[
            index_handler.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index,
                segment.y_index,
            )
        ] = -1 * (pv_to_absorber_conduction)

    solar_thermal_resultant_vector_absorbtion_term = (
        transmissivity_absorptivity_product(
            diffuse_reflection_coefficient=pvt_panel.glass.diffuse_reflection_coefficient,
            glass_transmissivity=pvt_panel.glass.transmissivity,
            layer_absorptivity=pvt_panel.pv.absorptivity,
        )
        * weather_conditions.irradiance  # [W/m^2]
        * segment.width  # [m]
        * segment.length  # [m]
        * (
            1
            - pvt_panel.pv.reference_efficiency
            * (
                1
                + pvt_panel.pv.thermal_coefficient * pvt_panel.pv.reference_temperature
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
        # Internal energy change
        + pv_internal_energy  # [W/K]
        * previous_temperature_vector[
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
) -> List[Tuple[List[float], float]]:
    """
    Returns matrix rows and resultant vector values representing system continuities.

    These inluce:
        - fluid entering the first section of the pipe is the same as that entering the
          collector (1);
        - fluid leaving the last section of the pipe is the same as that leaving the
          collector (2);
        - fluid entering the hot-water tank is the same as that leaving the collector
          (3);
        - fluid leaving the hot-water tank is the same as that entering the collector
          (4).

    :param number_of_temperatures:
        The number of temperatures being modelled in the system.

    :return:
        A `list` of `tuple`s containing:
        - the equation represented as a row in the matrix,
        - and the corresponding value in the resultant method.

    """

    equations: List[Tuple[List[float], float]] = list()

    # Equation 1: Continuity of fluid entering the collector.
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

    # Equation 2: Continuity of fluid leaving the collector.
    for pipe_number in range(number_of_pipes):
        row_equation = [0] * number_of_temperatures
        row_equation[
            index_handler.index_from_pipe_coordinates(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.htf_out,
                pipe_number,
                number_of_y_segments - 1,
            )
        ] = -1
        row_equation[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector_out,
            )
        ] = 1
        equations.append((row_equation, 0))

    # Equation 3: Fluid leaving the collector enters the tank without losses.
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
    best_guess_temperature_vector: numpy.ndarray,
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

    # Otherwise, the flow is diverted back into the collector.
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
    best_guess_temperature_vector: numpy.ndarray,
    heat_exchanger: exchanger.Exchanger,
    hot_water_load: float,
    hot_water_tank: tank.Tank,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
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

    # Compute the row equation
    row_equation: List[float] = [0] * number_of_temperatures

    if (
        previous_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank,
            )
        ]
        < 285
    ):
        import pdb

        pdb.set_trace()

    # Compute the T_t term
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.tank,
        )
    ] = (
        # Internal heat change
        hot_water_tank.mass  # [kg]
        * HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        / resolution  # [s]
        # Hot-water load
        + hot_water_load * HEAT_CAPACITY_OF_WATER  # [kg/s]  # [J/kg*K]
        # Heat loss
        + hot_water_tank.heat_loss_coefficient  # [W/kg*K]
        * hot_water_tank.area  # [m^2]
        # Heat input
        + (
            (
                pvt_panel.collector.mass_flow_rate  # [kg/s]
                * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
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
    )

    # Compute the T_c,out term
    row_equation[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector_out,
        )
    ] = -1 * (
        # Heat input
        (
            pvt_panel.collector.mass_flow_rate  # [kg/s]
            * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
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

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Internal heat change
        hot_water_tank.mass  # [kg]
        * HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        * previous_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank,
            )
        ]  # [K]
        / resolution  # [s]
        # Hot-water load.
        + hot_water_load  # [kg/s]
        * HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        * weather_conditions.mains_water_temperature  # [K]
        # Heat loss
        + hot_water_tank.heat_loss_coefficient  # [W/m^2*K]
        * hot_water_tank.area  # [m^2]
        * weather_conditions.ambient_tank_temperature  # [K]
    )

    return row_equation, resultant_vector_value


##################
# Public methods #
##################


def calculate_matrix_equation(
    best_guess_temperature_vector: numpy.ndarray,
    heat_exchanger: exchanger.Exchanger,
    hot_water_load: float,
    hot_water_tank: tank.Tank,
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Calculates and returns both the matrix and resultant vector for the matrix equation.

    :return:
        A `tuple` containing both the matrix "A" and resultant vector "B" for the matrix
        equation representing the temperature equations.

    """

    logger.debug("Matrix module called: calculating matrix and resultant vector.")

    # Instantiate an empty matrix and array based on the number of temperatures present.
    matrix = numpy.zeros([0, number_of_temperatures])
    resultant_vector = numpy.zeros([0, 1])

    # Calculate the glass equations.
    for segment in pvt_panel.segments.values():
        equation, resultant_value = _glass_equation(
            best_guess_temperature_vector,
            logger,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            previous_temperature_vector,
            pvt_panel,
            resolution,
            segment,
            weather_conditions,
        )
        logger.debug(
            "Glass equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    # Calculate the pv equations.
    for segment in pvt_panel.segments.values():
        equation, resultant_value = _pv_equation(
            best_guess_temperature_vector,
            logger,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            previous_temperature_vector,
            pvt_panel,
            resolution,
            segment,
            weather_conditions,
        )
        logger.debug(
            "PV equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    # Calculate the absorber-layer eqations.
    for segment in pvt_panel.segments.values():
        equation, resultant_value = _absorber_equation(
            logger,
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            previous_temperature_vector,
            pvt_panel,
            resolution,
            segment,
            weather_conditions,
        )
        logger.debug(
            "Collector equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    # Determine a sub-set of sections that have pipes attached.
    segments_with_pipes = [
        segment for segment in pvt_panel.segments.values() if segment.pipe
    ]
    if any([segment.pipe_index is None for segment in segments_with_pipes]):
        raise ProgrammerJudgementFault(
            "A segment specifies a pipe is present yet no pipe index is supplied. "
        )

    # Calculate the pipe equations.
    for segment in segments_with_pipes:
        equation, resultant_value = _pipe_equation(
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            previous_temperature_vector,
            pvt_panel,
            resolution,
            segment,
            weather_conditions,
        )
        logger.debug(
            "Pipe equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    # Calculate the htf equations.
    for segment in segments_with_pipes:
        equation, resultant_value = _htf_equation(
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            previous_temperature_vector,
            pvt_panel,
            resolution,
            segment,
        )
        logger.debug(
            "HTF equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    # Calculate the tank equations.
    equation, resultant_value = _tank_equation(
        best_guess_temperature_vector,
        heat_exchanger,
        hot_water_load,
        hot_water_tank,
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

    # Compute the HTF calculation equation.
    for segment in segments_with_pipes:
        equation, resultant_value = _htf_continuity_equation(
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            segment,
        )
        logger.debug(
            "HTF definition equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    # Compute the fluid continuity equations - there will be "N_y - 1" equations.
    for segment in [
        segment
        for segment in segments_with_pipes
        if segment.y_index < number_of_y_segments - 1
    ]:
        equation, resultant_value = _fluid_continuity_equation(
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            segment,
        )
        logger.debug(
            "Fluid continuity equation for segment %s computed:\nEquation: %s\nResultant value: %s W",
            segment.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    # Compute the system continuity equations and assign en masse.
    system_continuity_equations = _system_continuity_equations(
        number_of_pipes,
        number_of_temperatures,
        number_of_x_segments,
        number_of_y_segments,
    )

    for equation, resultant_value in system_continuity_equations:
        logger.debug(
            "System continuity equation computed:\nEquation: %s\nResultant value: %s W",
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))

    # Calculate the system boundary condition equations.
    boundary_condition_equations = _boundary_condition_equations(
        number_of_temperatures, number_of_x_segments, number_of_y_segments
    )

    for equation, resultant_value in boundary_condition_equations:
        logger.debug(
            "Boundary condition equation computed:\nEquation: %s\nResultant value: %s W",
            ", ".join([f"{value:.3f} W/K" for value in equation]),
            resultant_value,
        )
        matrix = numpy.vstack((matrix, equation))
        resultant_vector = numpy.vstack((resultant_vector, resultant_value))
        # if len(matrix) == number_of_temperatures:
        #     return matrix, resultant_vector

    return matrix, resultant_vector
