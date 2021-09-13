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

from tqdm import tqdm

from .. import exchanger, index_handler, tank
from ..pvt_collector import pvt

from ...__utils__ import (
    BColours,
    TemperatureName,
    OperatingMode,
    ProgrammerJudgementFault,
)
from ..__utils__ import WeatherConditions
from ..physics_utils import (
    convective_heat_transfer_coefficient_of_water,
    radiative_heat_transfer_coefficient,
)
from ..pvt_collector.element import Element
from ..pvt_collector.physics_utils import (
    glass_absorber_air_gap_resistance,
    glass_glass_air_gap_resistance,
    glass_pv_air_gap_resistance,
)

from .absorber import calculate_absorber_equation
from .continuity import (
    calculate_decoupled_system_continuity_equation,
    calculate_fluid_continuity_equation,
    calculate_system_continuity_equations,
)

from .glass import calculate_glass_equation
from .htf import calculate_htf_continuity_equation, calculate_htf_equation
from .pipe import calculate_pipe_equation
from .pv import calculate_pv_equation
from .tank import calculate_tank_continuity_equation, calculate_tank_equation
from .upper_glass import calculate_upper_glass_equation

__all__ = ("calculate_matrix_equation",)


####################
# Internal methods #
####################


def _boundary_condition_equations(
    number_of_temperatures: int,
    number_of_x_elements: int,
    number_of_y_elements: int,
    pvt_collector: pvt.PVT,
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
        for y_coord in range(number_of_y_elements):
            # Equation 1: Zero temperature gradient at x=0.
            row_equation: List[float] = [0] * number_of_temperatures
            row_equation[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    temperature_name,
                    0,
                    y_coord,
                )
            ] = -1
            row_equation[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    temperature_name,
                    1,
                    y_coord,
                )
            ] = 1
            equations.append((row_equation, 0))
            # Equation 2: Zero temperature gradient at x=W.
            row_equation = [0] * number_of_temperatures
            row_equation[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    temperature_name,
                    number_of_x_elements - 1,
                    y_coord,
                )
            ] = -1
            row_equation[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    temperature_name,
                    number_of_x_elements - 2,
                    y_coord,
                )
            ] = 1
            equations.append((row_equation, 0))

        # Work along both "top" and "bottom" edges, applying the boundary conditions.
        for x_coord in range(number_of_x_elements):
            # Equation 3: Zero temperature gradient at y=0.
            row_equation = [0] * number_of_temperatures
            row_equation[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    temperature_name,
                    x_coord,
                    0,
                )
            ] = -1
            row_equation[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    temperature_name,
                    x_coord,
                    1,
                )
            ] = 1
            equations.append((row_equation, 0))
            # Equation 4: Zero temperature gradient at y=H.
            row_equation = [0] * number_of_temperatures
            row_equation[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    temperature_name,
                    x_coord,
                    number_of_y_elements - 1,
                )
            ] = -1
            row_equation[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    temperature_name,
                    x_coord,
                    number_of_y_elements - 2,
                )
            ] = 1
            equations.append((row_equation, 0))

    return equations


def _calculate_downward_glass_terms(
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    element: Element,
    logger: logging.Logger,
    number_of_x_elements: int,
    pvt_collector: pvt.PVT,
    weather_conditions: WeatherConditions,
) -> Tuple[float, float]:
    """
    Calculates the downward glass terms.

    :return:
        A `tuple` containing:
        - the downward conduction from the glass layer to the layer below,
        - the downward radiation from the glass layer to the layer below.

    """

    # If the PV layer is below the glass layer.
    if element.glass and element.pv:
        if pvt_collector.glass is None:
            raise ProgrammerJudgementFault(
                "{}Element {} has a glass layer but no glass data supplied ".format(
                    BColours.FAIL, element
                )
                + "in the PV-T data file.{}".format(BColours.ENDC)
            )

        glass_downward_conduction = (
            element.width
            * element.length
            / glass_pv_air_gap_resistance(
                pvt_collector,
                0.5
                * (
                    best_guess_temperature_vector[
                        index_handler.index_from_element_coordinates(
                            number_of_x_elements,
                            pvt_collector,
                            TemperatureName.pv,
                            element.x_index,
                            element.y_index,
                        )
                    ]
                    + best_guess_temperature_vector[
                        index_handler.index_from_element_coordinates(
                            number_of_x_elements,
                            pvt_collector,
                            TemperatureName.glass,
                            element.x_index,
                            element.y_index,
                        )
                    ]
                ),
                weather_conditions,
            )
        )
        logger.debug("Glass to pv conduction %s W/K", glass_downward_conduction)

        glass_downward_radiation = (
            element.width
            * element.length
            * radiative_heat_transfer_coefficient(
                destination_emissivity=pvt_collector.pv.emissivity,
                destination_temperature=best_guess_temperature_vector[
                    index_handler.index_from_element_coordinates(
                        number_of_x_elements,
                        pvt_collector,
                        TemperatureName.pv,
                        element.x_index,
                        element.y_index,
                    )
                ],
                source_emissivity=pvt_collector.glass.emissivity,
                source_temperature=best_guess_temperature_vector[
                    index_handler.index_from_element_coordinates(
                        number_of_x_elements,
                        pvt_collector,
                        TemperatureName.glass,
                        element.x_index,
                        element.y_index,
                    )
                ],
            )
        )
        logger.debug("Glass to pv radiation %s W/K", glass_downward_radiation)

        return glass_downward_conduction, glass_downward_radiation

    # If the absorber layer is below the glass layer.
    if element.glass and element.absorber:
        if pvt_collector.glass is None:
            raise ProgrammerJudgementFault(
                "{}Element {} has a glass layer but no glass data supplied ".format(
                    BColours.FAIL, element
                )
                + "in the PV-T data file.{}".format(BColours.ENDC)
            )

        glass_downward_conduction = (
            element.width
            * element.length
            / glass_absorber_air_gap_resistance(
                pvt_collector,
                0.5
                * (
                    best_guess_temperature_vector[
                        index_handler.index_from_element_coordinates(
                            number_of_x_elements,
                            pvt_collector,
                            TemperatureName.absorber,
                            element.x_index,
                            element.y_index,
                        )
                    ]
                    + best_guess_temperature_vector[
                        index_handler.index_from_element_coordinates(
                            number_of_x_elements,
                            pvt_collector,
                            TemperatureName.glass,
                            element.x_index,
                            element.y_index,
                        )
                    ]
                ),
                weather_conditions,
            )
        )
        logger.debug("Glass to absorber conduction %s W/K", glass_downward_conduction)

        glass_downward_radiation = (
            element.width
            * element.length
            * radiative_heat_transfer_coefficient(
                destination_emissivity=pvt_collector.absorber.emissivity,
                destination_temperature=best_guess_temperature_vector[
                    index_handler.index_from_element_coordinates(
                        number_of_x_elements,
                        pvt_collector,
                        TemperatureName.absorber,
                        element.x_index,
                        element.y_index,
                    )
                ],
                source_emissivity=pvt_collector.glass.emissivity,
                source_temperature=best_guess_temperature_vector[
                    index_handler.index_from_element_coordinates(
                        number_of_x_elements,
                        pvt_collector,
                        TemperatureName.glass,
                        element.x_index,
                        element.y_index,
                    )
                ],
            )
        )
        logger.debug("Glass to absorber radiation %s W/K", glass_downward_radiation)

        return glass_downward_conduction, glass_downward_radiation

    # Otherwise, there is no glass layer, so these terms are zero.
    logger.debug(
        "No glass-to-pv conduction or radiation because {} ".format(
            " and ".join(
                {
                    entry
                    for entry in {
                        "glass" if not element.glass else None,
                        "pv" if not element.pv else None,
                    }
                    if entry is not None
                }
            )
        )
        + "layer(s) not present."
    )
    return 0, 0


def _calculate_downward_upper_glass_terms(
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    element: Element,
    logger: logging.Logger,
    number_of_x_elements: int,
    pvt_collector: pvt.PVT,
    weather_conditions: WeatherConditions,
) -> Tuple[float, float]:
    """
    Calculates the downward conductive and radiative terms between the glass layers.

    :return:
        A `tuple` containing:
        - the conductive term between the two glass layers,
        - the radiative term between the two glass layers.

    """

    # If there is no double glazing, then these terms are zero.
    if not element.upper_glass:
        return 0, 0

    if not pvt_collector.upper_glass:
        raise ProgrammerJudgementFault(
            "{}Cannot compute double-glazing without `upper_glass` layer ".format(
                BColours.FAIL
            )
            + "present.{}".format(BColours.ENDC)
        )

    if not pvt_collector.glass:
        raise ProgrammerJudgementFault(
            "{}Cannot compute double-glazing without `glass` layer present.{}".format(
                BColours.FAIL, BColours.ENDC
            )
        )

    # Otherwise, calculate the terms.
    upper_glass_downward_conduction = (
        element.width
        * element.length
        / glass_glass_air_gap_resistance(
            pvt_collector,
            0.5
            * (
                best_guess_temperature_vector[
                    index_handler.index_from_element_coordinates(
                        number_of_x_elements,
                        pvt_collector,
                        TemperatureName.upper_glass,
                        element.x_index,
                        element.y_index,
                    )
                ]
                + best_guess_temperature_vector[
                    index_handler.index_from_element_coordinates(
                        number_of_x_elements,
                        pvt_collector,
                        TemperatureName.glass,
                        element.x_index,
                        element.y_index,
                    )
                ]
            ),
            weather_conditions,
        )
    )
    logger.debug(
        "Upper glass to glass conduction %s W/K", upper_glass_downward_conduction
    )

    upper_glass_downward_radiation = (
        element.width
        * element.length
        * radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_collector.glass.emissivity,
            destination_temperature=best_guess_temperature_vector[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    TemperatureName.glass,
                    element.x_index,
                    element.y_index,
                )
            ],
            source_emissivity=pvt_collector.upper_glass.emissivity,
            source_temperature=best_guess_temperature_vector[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    TemperatureName.upper_glass,
                    element.x_index,
                    element.y_index,
                )
            ],
        )
    )
    logger.debug(
        "Upper glass to glass radiation %s W/K", upper_glass_downward_radiation
    )

    return upper_glass_downward_conduction, upper_glass_downward_radiation


##################
# Public methods #
##################


def calculate_matrix_equation(  # pylint: disable=too-many-branches
    *,
    best_guess_temperature_vector: Union[List[float], numpy.ndarray],
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    number_of_y_elements: int,
    operating_mode: OperatingMode,
    pvt_collector: pvt.PVT,
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

    for element_coordinates, element in tqdm(
        pvt_collector.elements.items(),
        desc="computing matrix equation",
        unit="element",
        leave=False,
    ):
        logger.debug("Calculating equations for element %s", element_coordinates)
        # Compute the various shared values.
        (
            upper_glass_downward_conduction,
            upper_glass_downward_radiation,
        ) = _calculate_downward_upper_glass_terms(
            best_guess_temperature_vector,
            element,
            logger,
            number_of_x_elements,
            pvt_collector,
            weather_conditions,
        )

        (
            glass_downward_conduction,
            glass_downward_radiation,
        ) = _calculate_downward_glass_terms(
            best_guess_temperature_vector,
            element,
            logger,
            number_of_x_elements,
            pvt_collector,
            weather_conditions,
        )

        pv_to_absorber_conduction = (
            element.width
            * element.length
            / pvt_collector.pv_to_absorber_thermal_resistance
        )
        logger.debug("PV to absorber conduction: %s W/K", pv_to_absorber_conduction)

        absorber_to_pipe_conduction = (
            (
                element.width  # [m]
                * element.length  # [m]
                * pvt_collector.bond.conductivity  # [W/m*K]
                / pvt_collector.bond.thickness  # [m]
            )
            if element.pipe
            else 0
        )
        logger.debug("Absorber to pipe conduction: %s W/K", absorber_to_pipe_conduction)

        if element.upper_glass:
            (
                upper_glass_equation,
                upper_glass_resultant_value,
            ) = calculate_upper_glass_equation(
                best_guess_temperature_vector,
                logger,
                number_of_temperatures,
                number_of_x_elements,
                number_of_y_elements,
                operating_mode,
                previous_temperature_vector,
                pvt_collector,
                resolution,
                element,
                upper_glass_downward_conduction,
                upper_glass_downward_radiation,
                weather_conditions,
            )
            logger.debug(
                "Upper glass equation for element %s computed:\nEquation: %s\n"
                "Resultant value: %s W",
                element.coordinates,
                ", ".join([f"{value:.3f} W/K" for value in upper_glass_equation]),
                upper_glass_resultant_value,
            )
            matrix = numpy.vstack((matrix, upper_glass_equation))
            resultant_vector = numpy.vstack(
                (resultant_vector, upper_glass_resultant_value)
            )

        if element.glass:
            glass_equation, glass_resultant_value = calculate_glass_equation(
                best_guess_temperature_vector,
                glass_downward_conduction,
                glass_downward_radiation,
                logger,
                number_of_temperatures,
                number_of_x_elements,
                number_of_y_elements,
                operating_mode,
                previous_temperature_vector,
                pvt_collector,
                resolution,
                element,
                upper_glass_downward_conduction,
                upper_glass_downward_radiation,
                weather_conditions,
            )
            logger.debug(
                "Glass equation for element %s computed:\nEquation: %s\nResultant value: %s W",
                element.coordinates,
                ", ".join([f"{value:.3f} W/K" for value in glass_equation]),
                glass_resultant_value,
            )
            matrix = numpy.vstack((matrix, glass_equation))
            resultant_vector = numpy.vstack((resultant_vector, glass_resultant_value))

        if element.pv:
            pv_equation, pv_resultant_value = calculate_pv_equation(
                best_guess_temperature_vector,
                logger,
                number_of_temperatures,
                number_of_x_elements,
                number_of_y_elements,
                operating_mode,
                previous_temperature_vector,
                pv_to_absorber_conduction,
                glass_downward_conduction,
                glass_downward_radiation,
                pvt_collector,
                resolution,
                element,
                weather_conditions,
            )
            logger.debug(
                "PV equation for element %s computed:\nEquation: %s\nResultant value: %s W",
                element.coordinates,
                ", ".join([f"{value:.3f} W/K" for value in pv_equation]),
                pv_resultant_value,
            )
            matrix = numpy.vstack((matrix, pv_equation))
            resultant_vector = numpy.vstack((resultant_vector, pv_resultant_value))

        if element.absorber:
            absorber_equation, absorber_resultant_value = calculate_absorber_equation(
                absorber_to_pipe_conduction,
                best_guess_temperature_vector,
                glass_downward_conduction,
                glass_downward_radiation,
                logger,
                number_of_pipes,
                number_of_temperatures,
                number_of_x_elements,
                number_of_y_elements,
                operating_mode,
                previous_temperature_vector,
                pv_to_absorber_conduction,
                pvt_collector,
                resolution,
                element,
                weather_conditions,
            )
            logger.debug(
                "Collector equation for element %s computed:\nEquation: %s\nResultant value: %s W",
                element.coordinates,
                ", ".join([f"{value:.3f} W/K" for value in absorber_equation]),
                absorber_resultant_value,
            )
            matrix = numpy.vstack((matrix, absorber_equation))
            resultant_vector = numpy.vstack(
                (resultant_vector, absorber_resultant_value)
            )

        # Only calculate the pipe equations if the element has an associated pipe.
        if not element.pipe:
            logger.debug("3 equations for element %s", element_coordinates)
            continue

        pipe_to_htf_heat_transfer = (
            element.length  # [m]
            * numpy.pi
            * pvt_collector.absorber.inner_pipe_diameter  # [m]
            * convective_heat_transfer_coefficient_of_water(
                best_guess_temperature_vector[
                    index_handler.index_from_pipe_coordinates(
                        number_of_pipes,
                        number_of_x_elements,
                        element.pipe_index,  # type: ignore
                        pvt_collector,
                        TemperatureName.htf,
                        element.y_index,
                    )
                ],
                pvt_collector,
                weather_conditions,
            )  # [W/m^2*K]
        )
        logger.debug("Pipe to HTF heat transfer: %s W/K", pipe_to_htf_heat_transfer)

        pipe_equation, pipe_resultant_value = calculate_pipe_equation(
            absorber_to_pipe_conduction,
            best_guess_temperature_vector,
            logger,
            number_of_pipes,
            number_of_temperatures,
            number_of_x_elements,
            operating_mode,
            pipe_to_htf_heat_transfer,
            previous_temperature_vector,
            pvt_collector,
            resolution,
            element,
            weather_conditions,
        )
        logger.debug(
            "Pipe equation for element %s computed:\nEquation: %s\nResultant value: %s W",
            element.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in pipe_equation]),
            pipe_resultant_value,
        )
        matrix = numpy.vstack((matrix, pipe_equation))
        resultant_vector = numpy.vstack((resultant_vector, pipe_resultant_value))

        htf_equation, htf_resultant_value = calculate_htf_equation(
            best_guess_temperature_vector,
            number_of_pipes,
            number_of_temperatures,
            number_of_x_elements,
            operating_mode,
            pipe_to_htf_heat_transfer,
            previous_temperature_vector,
            pvt_collector,
            resolution,
            element,
        )
        logger.debug(
            "HTF equation for element %s computed:\nEquation: %s\nResultant value: %s W",
            element.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in htf_equation]),
            htf_resultant_value,
        )
        matrix = numpy.vstack((matrix, htf_equation))
        resultant_vector = numpy.vstack((resultant_vector, htf_resultant_value))

        htf_equation, htf_resultant_value = calculate_htf_continuity_equation(
            number_of_pipes,
            number_of_temperatures,
            number_of_x_elements,
            pvt_collector,
            element,
        )
        logger.debug(
            "HTF definition equation for element %s computed:\nEquation: %s\nResultant value: %s W",
            element.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in htf_equation]),
            htf_resultant_value,
        )
        matrix = numpy.vstack((matrix, htf_equation))
        resultant_vector = numpy.vstack((resultant_vector, htf_resultant_value))

        # Fluid continuity equations only need to be computed if there exist multiple
        # connected elements.
        if element.y_index >= number_of_y_elements - 1:
            logger.debug("6 equations for element %s", element_coordinates)
            continue

        (
            fluid_continuity_equation,
            fluid_continuity_resultant_value,
        ) = calculate_fluid_continuity_equation(
            number_of_pipes,
            number_of_temperatures,
            number_of_x_elements,
            pvt_collector,
            element,
        )
        logger.debug(
            "Fluid continuity equation for element %s computed:\nEquation: %s\n"
            "Resultant value: %s W",
            element.coordinates,
            ", ".join([f"{value:.3f} W/K" for value in fluid_continuity_equation]),
            fluid_continuity_resultant_value,
        )
        matrix = numpy.vstack((matrix, fluid_continuity_equation))
        resultant_vector = numpy.vstack(
            (resultant_vector, fluid_continuity_resultant_value)
        )
        logger.debug("7 equations for element %s", element_coordinates)

    # # Calculate the system boundary condition equations.
    # boundary_condition_equations = _boundary_condition_equations(
    #     number_of_temperatures, number_of_x_elements, number_of_y_elements
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
        decoupled_system_continuity_equations = (
            calculate_decoupled_system_continuity_equation(
                collector_input_temperature,
                number_of_pipes,
                number_of_temperatures,
                number_of_x_elements,
                number_of_y_elements,
                pvt_collector,
            )
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
    equation, resultant_value = calculate_tank_equation(
        best_guess_temperature_vector,
        heat_exchanger,
        hot_water_load,
        hot_water_tank,
        logger,
        number_of_temperatures,
        previous_temperature_vector,
        pvt_collector,
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
    equation, resultant_value = calculate_tank_continuity_equation(
        best_guess_temperature_vector,
        heat_exchanger,
        number_of_temperatures,
        pvt_collector,
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
    system_continuity_equations = calculate_system_continuity_equations(
        number_of_pipes,
        number_of_temperatures,
        number_of_x_elements,
        number_of_y_elements,
        previous_temperature_vector,
        pvt_collector,
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
