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
from .pvt_panel.segment import Segment

from ..__utils__ import TemperatureName, ProgrammerJudgementFault
from .__utils__ import WeatherConditions
from .constants import STEFAN_BOLTZMAN_CONSTANT
from .physics_utils import (
    radiative_heat_transfer_coefficient,
    transmissivity_absorptivity_product,
)

__all__ = ("calculate_matrix_equation",)


####################
# Internal methods #
####################


def _absorber_equation(
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: Tuple[float, ...],
    pvt_panel: pvt.PVT,
    resolution: int,
    segment: Segment,
    weather_conditions: WeatherConditions,
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

    # Compute the row equation
    row_equation = [0] * number_of_temperatures

    # Compute the T_A(i, j) term
    row_equation[
        index.index_from_segment_coordinates(
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector,
            segment.x_index,
            segment.y_index,
        )
    ] = (
        # Internal heat change.
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.collector.thickness  # [m]
        * pvt_panel.collector.density  # [kg/m^3]
        * pvt_panel.collector.heat_capacity  # [J/kg*K]
        / resolution  # [s]
        # X-wise conduction within the glass layer
        + (2 if segment.x_index not in [0, number_of_x_segments - 1] else 1)
        * pvt_panel.collector.conductivity  # [W/m*K]
        * pvt_panel.collector.thickness  # [m]
        * segment.length  # [m]
        / segment.width  # [m]
        # Y-wise conduction within the glass layer
        + (2 if segment.y_index not in [0, number_of_y_segments - 1] else 1)
        * pvt_panel.collector.conductivity  # [W/m*K]
        * pvt_panel.collector.thickness  # [m]
        * segment.width  # [m]
        / segment.length  # [m]
        # Conduction from the PV layer
        + segment.width * segment.length / pvt_panel.pv_to_collector_thermal_resistance
        # Conduction to the pipe, if present.
        + (
            segment.width  # [m]
            * segment.length  # [m]
            * pvt_panel.bond.conductivity  # [W/m*K]
            / pvt_panel.bond.thickness  # [m]
        )
        if segment.pipe
        else 0
        # Loss through the back if no pipe is present.
        + (segment.width * segment.length * pvt_panel.insulation_thermal_resistance)
        if not segment.pipe
        else 0
    )

    # Compute the T_A(i+1, j) term provided that that segment exists.
    if segment.x_index + 1 < number_of_x_segments:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index + 1,
                segment.y_index,
            )
        ] = (
            -1
            * pvt_panel.collector.conductivity  # [W/m*K]
            * pvt_panel.collector.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_A(i-1, j) term provided that that segment exists.
    if segment.x_index > 0:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index - 1,
                segment.y_index,
            )
        ] = (
            -1
            * pvt_panel.collector.conductivity  # [W/m*K]
            * pvt_panel.collector.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_A(i, j+1) term provided that that segment exists.
    if segment.y_index + 1 < number_of_y_segments:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index,
                segment.y_index + 1,
            )
        ] = (
            -1
            * pvt_panel.collector.conductivity  # [W/m*K]
            * pvt_panel.collector.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_A(i, j-1) term provided that that segment exists.
    if segment.y_index > 0:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index,
                segment.y_index - 1,
            )
        ] = (
            -1
            * pvt_panel.collector.conductivity  # [W/m*K]
            * pvt_panel.collector.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_pv(i, j) term provided that there is a collector layer present.
    if segment.pv:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index,
                segment.y_index,
            )
        ] = -1 * (
            segment.width  # [m]
            * segment.length  # [m]
            / pvt_panel.pv_to_collector_thermal_resistance
        )

    # Compute the T_P(pipe_number, j) term provided that there is a pipe present.
    if segment.pipe:
        try:
            row_equation[
                index.index_from_pipe_coordinates(
                    number_of_pipes, TemperatureName.pipe, segment.pipe_index
                )
            ] = (
                segment.width  # [m]
                * segment.length  # [m]
                * pvt_panel.bond.conductivity  # [W/m*K]
                / pvt_panel.bond.thickness  # [m]
            )
        except ProgrammerJudgementFault as e:
            raise ProgrammerJudgementFault(
                "Error determining pipe temperature term. "
                f"Likely that pipe index was not specified: {str(e)}"
            ) from None

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Internal heat change term.
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.collector.thickness  # [m]
        * pvt_panel.collector.density  # [kg/m^3]
        * pvt_panel.collector.heat_capacity  # [J/kg*K]
        * previous_temperature_vector[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index,
                segment.y_index,
            )
        ]
        / resolution  # [s]
        # Ambient temperature term.
        + (
            segment.width  # [m]
            * segment.length  # [m]
            * weather_conditions.ambient_temperature  # [K]
            / pvt_panel.insulation_thermal_resistance  # [m^2*K/W]
        )
        if not segment.pipe
        else 0
    )

    return row_equation, resultant_vector_value


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
    best_guess_temperature_vector: Tuple[float, ...],
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: Tuple[float, ...],
    pvt_panel: pvt.PVT,
    resolution: int,
    segment: Segment,
    weather_conditions: WeatherConditions,
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

    # Compute the row equation
    row_equation = [0] * number_of_temperatures

    # Compute the T_g(i, j) term
    row_equation[
        index.index_from_segment_coordinates(
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.glass,
            segment.x_index,
            segment.y_index,
        )
    ] = (
        # Internal heat change.
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.glass.thickness  # [m]
        * pvt_panel.glass.density  # [kg/m^3]
        * pvt_panel.glass.heat_capacity  # [J/kg*K]
        / resolution  # [s]
        # X-wise conduction within the glass layer
        + (2 if segment.x_index not in [0, number_of_x_segments - 1] else 1)
        * pvt_panel.glass.conductivity  # [W/m*K]
        * pvt_panel.glass.thickness  # [m]
        * segment.length  # [m]
        / segment.width  # [m]
        # Y-wise conduction within the glass layer
        + (2 if segment.y_index not in [0, number_of_y_segments - 1] else 1)
        * pvt_panel.glass.conductivity  # [W/m*K]
        * pvt_panel.glass.thickness  # [m]
        * segment.width  # [m]
        / segment.length  # [m]
        # Conduction to the air.
        + segment.width  # [m]
        * segment.length  # [m]
        * weather_conditions.wind_heat_transfer_coefficient  # [W/m^2*K]
        # Radiation to the sky.
        + segment.width  # [m]
        * segment.length  # [m]
        * radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=best_guess_temperature_vector[
                index.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.glass,
                    segment.x_index,
                    segment.y_index,
                )
            ],
        )
        # Radiation to the PV layer
        + segment.width
        * segment.length
        * radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.pv.emissivity,
            destination_temperature=best_guess_temperature_vector[
                index.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.pv,
                    segment.x_index,
                    segment.y_index,
                )
            ],
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=best_guess_temperature_vector[
                index.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.glass,
                    segment.x_index,
                    segment.y_index,
                )
            ],
        )
        # Conduction to the PV layer
        + segment.width * segment.length / pvt_panel.air_gap_resistance
    )

    # Compute the T_g(i+1, j) term provided that that segment exists.
    if segment.x_index + 1 < number_of_x_segments:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index + 1,
                segment.y_index,
            )
        ] = (
            -1
            * pvt_panel.glass.conductivity  # [W/m*K]
            * pvt_panel.glass.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_g(i-1, j) term provided that that segment exists.
    if segment.x_index > 0:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index - 1,
                segment.y_index,
            )
        ] = (
            -1
            * pvt_panel.glass.conductivity  # [W/m*K]
            * pvt_panel.glass.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_g(i, j+1) term provided that that segment exists.
    if segment.y_index + 1 < number_of_y_segments:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index,
                segment.y_index + 1,
            )
        ] = (
            -1
            * pvt_panel.glass.conductivity  # [W/m*K]
            * pvt_panel.glass.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_g(i, j-1) term provided that that segment exists.
    if segment.y_index > 0:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.glass,
                segment.x_index,
                segment.y_index - 1,
            )
        ] = (
            -1
            * pvt_panel.glass.conductivity  # [W/m*K]
            * pvt_panel.glass.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_pv(i, j) term provided that there is a PV layer present.
    if segment.pv:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index,
                segment.y_index,
            )
        ] = -1 * (
            segment.width  # [m]
            * segment.length  # [m]
            * (
                # Radiative term
                radiative_heat_transfer_coefficient(
                    destination_emissivity=pvt_panel.pv.emissivity,
                    destination_temperature=best_guess_temperature_vector[
                        index.index_from_segment_coordinates(
                            number_of_x_segments,
                            number_of_y_segments,
                            TemperatureName.pv,
                            segment.x_index,
                            segment.y_index,
                        )
                    ],
                    source_emissivity=pvt_panel.glass.emissivity,
                    source_temperature=best_guess_temperature_vector[
                        index.index_from_segment_coordinates(
                            number_of_x_segments,
                            number_of_y_segments,
                            TemperatureName.glass,
                            segment.x_index,
                            segment.y_index,
                        )
                    ],
                )
                # Conductive term
                + 1 / pvt_panel.air_gap_resistance
            )
        )

    # Compute the resultant vector value.
    resultant_vector_value = (
        # Previous glass temperature term.
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.glass.thickness  # [m]
        * pvt_panel.glass.density  # [kg/m^3]
        * pvt_panel.glass.heat_capacity  # [J/kg*K]
        * previous_temperature_vector[
            index.index_from_segment_coordinates(
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
        * weather_conditions.ambient_temperature
        # Sky temperature term.
        + segment.width  # [m]
        * segment.length  # [m]
        * radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=best_guess_temperature_vector[
                index.index_from_segment_coordinates(
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
    best_guess_temperature_vector: Tuple[float, ...],
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_temperature_vector: Tuple[float, ...],
    pvt_panel: pvt.PVT,
    resolution: int,
    segment: Segment,
    weather_conditions: WeatherConditions,
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

    # Compute the row equation
    row_equation = [0] * number_of_temperatures

    # Compute the T_pv(i, j) term
    row_equation[
        index.index_from_segment_coordinates(
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.pv,
            segment.x_index,
            segment.y_index,
        )
    ] = (
        # Internal heat change.
        segment.width  # [m]
        * segment.length  # [m]
        * pvt_panel.pv.thickness  # [m]
        * pvt_panel.pv.density  # [kg/m^3]
        * pvt_panel.pv.heat_capacity  # [J/kg*K]
        / resolution  # [s]
        # X-wise conduction within the glass layer
        + (2 if segment.x_index not in [0, number_of_x_segments - 1] else 1)
        * pvt_panel.pv.conductivity  # [W/m*K]
        * pvt_panel.pv.thickness  # [m]
        * segment.length  # [m]
        / segment.width  # [m]
        # Y-wise conduction within the glass layer
        + (2 if segment.y_index not in [0, number_of_y_segments - 1] else 1)
        * pvt_panel.pv.conductivity  # [W/m*K]
        * pvt_panel.pv.thickness  # [m]
        * segment.width  # [m]
        / segment.length  # [m]
        # Radiation to the glass layer
        + segment.width
        * segment.length
        * radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=best_guess_temperature_vector[
                index.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.glass,
                    segment.x_index,
                    segment.y_index,
                )
            ],
            source_emissivity=pvt_panel.pv.emissivity,
            source_temperature=best_guess_temperature_vector[
                index.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.pv,
                    segment.x_index,
                    segment.y_index,
                )
            ],
        )
        # Conduction to the glass layer
        + segment.width * segment.length / pvt_panel.air_gap_resistance
        # Conduction to the absorber layer
        + segment.width * segment.length / pvt_panel.pv_to_collector_thermal_resistance
        # Solar thermal absorption
        - transmissivity_absorptivity_product(
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

    # Compute the T_pv(i+1, j) term provided that that segment exists.
    if segment.x_index + 1 < number_of_x_segments:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index + 1,
                segment.y_index,
            )
        ] = (
            -1
            * pvt_panel.pv.conductivity  # [W/m*K]
            * pvt_panel.pv.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_pv(i-1, j) term provided that that segment exists.
    if segment.x_index > 0:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index - 1,
                segment.y_index,
            )
        ] = (
            -1
            * pvt_panel.pv.conductivity  # [W/m*K]
            * pvt_panel.pv.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_pv(i, j+1) term provided that that segment exists.
    if segment.y_index + 1 < number_of_y_segments:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index,
                segment.y_index + 1,
            )
        ] = (
            -1
            * pvt_panel.pv.conductivity  # [W/m*K]
            * pvt_panel.pv.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_pv(i, j-1) term provided that that segment exists.
    if segment.y_index > 0:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index,
                segment.y_index - 1,
            )
        ] = (
            -1
            * pvt_panel.pv.conductivity  # [W/m*K]
            * pvt_panel.pv.thickness  # [m]
            * segment.height
            / segment.width
        )

    # Compute the T_g(i, j) term provided that there is a glass layer present.
    if segment.glass:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.pv,
                segment.x_index,
                segment.y_index,
            )
        ] = -1 * (
            segment.width  # [m]
            * segment.length  # [m]
            * (
                # Radiative term
                radiative_heat_transfer_coefficient(
                    destination_emissivity=pvt_panel.glass.emissivity,
                    destination_temperature=best_guess_temperature_vector[
                        index.index_from_segment_coordinates(
                            number_of_x_segments,
                            number_of_y_segments,
                            TemperatureName.glass,
                            segment.x_index,
                            segment.y_index,
                        )
                    ],
                    source_emissivity=pvt_panel.pv.emissivity,
                    source_temperature=best_guess_temperature_vector[
                        index.index_from_segment_coordinates(
                            number_of_x_segments,
                            number_of_y_segments,
                            TemperatureName.pv,
                            segment.x_index,
                            segment.y_index,
                        )
                    ],
                )
                # Conductive term
                + 1 / pvt_panel.air_gap_resistance
            )
        )

    # Compute the T_A(i, j) term provided that there is a collector layer present.
    if segment.collector:
        row_equation[
            index.index_from_segment_coordinates(
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector,
                segment.x_index,
                segment.y_index,
            )
        ] = -1 * (
            segment.width  # [m]
            * segment.length  # [m]
            / pvt_panel.pv_to_collector_thermal_resistance
        )

    # Compute the resultant vector value.
    resultant_vector_value = (
        transmissivity_absorptivity_product(
            diffuse_reflection_coefficient=pvt_panel.glass.diffuse_reflection_coefficient,
            glass_transmissivity=pvt_panel.glass.transmissivity,
            layer_absorptivity=pvt_panel.pv.absorptivity,
        )
        * weather_conditions.irradiance
        # [W/m^2]
        + segment.width  # [m]
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

    return row_equation, resultant_vector_value


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


def calculate_matrix_equation(
    best_guess_temperature_vector: Tuple[float, ...],
    previous_temperature_vector: Tuple[float, ...],
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,
) -> Tuple[numpy.ndarray, Tuple[float, ...]]:
    """
    Calculates and returns both the matrix and resultant vector for the matrix equation.

    :return:
        A `tuple` containing both the matrix "A" and resultant vector "B" for the matrix
        equation representing the temperature equations.

    """

    # Set up an index for tracking the equation number.
    equation_index: int = 0

    # Determine the number of temperatures being modelled.
    number_of_pipes = len({segment.pipe_index for segment in pvt_panel.segments})
    number_of_x_segments = len({segment.x_index for segment in pvt_panel.segments})
    number_of_y_segments = len({segment.y_index for segment in pvt_panel.segments})
    number_of_temperatures = index.num_temperatures(
        number_of_pipes, number_of_x_segments, number_of_y_segments
    )

    # Instantiate an empty matrix and array based on the number of temperatures present.
    matrix = numpy.zeros([number_of_temperatures, number_of_temperatures])
    reslutant_vector = numpy.zeros([number_of_temperatures, 1])

    # * Iterate through and generate...

    # Calculate the glass equations.
    for segment in pvt_panel.segments:
        matrix[equation_index], reslutant_vector[equation_index] = _glass_equation(
            best_guess_temperature_vector,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            previous_temperature_vector,
            pvt_panel,
            resolution,
            segment,
            weather_conditions,
        )
        equation_index += 1

    # Calculate the pv equations.
    for segment in pvt_panel.segments:
        matrix[equation_index], reslutant_vector[equation_index] = _pv_equation(
            best_guess_temperature_vector,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            previous_temperature_vector,
            pvt_panel,
            resolution,
            segment,
            weather_conditions,
        )
        equation_index += 1

    # Calculate the absorber-layer eqations.

    # * and the pipe equations;

    # * the htf equations,

    # * htf input equations,

    # * and htf output equations,

    # * and the various continuity equations,

    # * along with the tank equation.
