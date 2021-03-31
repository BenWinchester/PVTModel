#!/usr/bin/python3.7
########################################################################################
# __main__.py - The main module for this, my first, PV-T model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The main module for the PV-T model.

This module coordinates the time-stepping of the itterative module, calling out to the
various modules where necessary, as well as reading in the various data files and
processing the command-line arguments that define the scope of the model run.

"""

import datetime
import logging
import os

from typing import Dict, List, Optional, Set, Tuple

import numpy
import pytz

from dateutil.relativedelta import relativedelta
from scipy import linalg  # type: ignore

from . import (
    efficiency,
    exchanger,
    index_handler,
    load,
    mains_power,
    matrix,
    process_pvt_system_data,
    tank,
    weather,
)

from .pvt_panel import pvt

from ..__utils__ import (  # pylint: disable=unused-import
    BColours,
    CarbonEmissions,
    get_logger,
    OperatingMode,
    SystemData,
    TemperatureName,
    TotalPowerData,
)
from .__utils__ import (
    DivergentSolutionError,
    ProgrammerJudgementFault,
    PVT_SYSTEM_MODEL_LOGGER_NAME,
    time_iterator,
    WeatherConditions,
)

from .constants import (
    COLLECTOR_INPUT_TEMPERATURES,
    CONVERGENT_SOLUTION_PRECISION,
    MAXIMUM_RECURSION_DEPTH,
    WARN_RECURSION_DEPTH,
    ZERO_CELCIUS_OFFSET,
)

from .pvt_panel.segment import Segment, SegmentCoordinates

# The temperature of hot-water required by the end-user, measured in Kelvin.
HOT_WATER_DEMAND_TEMP = 60 + ZERO_CELCIUS_OFFSET
# The initial date and time for the simultion to run from.
DEFAULT_INITIAL_DATE_AND_TIME = datetime.datetime(2005, 1, 1, 0, 0, tzinfo=pytz.UTC)
# The average temperature of the air surrounding the tank, which is internal to the
# household, measured in Kelvin.
INTERNAL_HOUSEHOLD_AMBIENT_TEMPERATURE = ZERO_CELCIUS_OFFSET + 20  # [K]
# Folder containing the solar irradiance profiles
SOLAR_IRRADIANCE_FOLDERNAME = "solar_irradiance_profiles"
# Folder containing the temperature profiles
TEMPERATURE_FOLDERNAME = "temperature_profiles"
# Name of the weather data file.
WEATHER_DATA_FILENAME = "weather.yaml"


def _average_layer_temperature(
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    temperature_name: TemperatureName,
    temperature_vector: numpy.ndarray,
) -> float:
    """
    Determines the average temperature for a layer.

    :param number_of_pipes:
        The number of pipes in the absorber.

    :param number_of_x_segments:
        The number of segments in a single row of the absorber.

    :param number_of_y_segments:
        The number of y segments in a single row of the absorber.

    :param temperature_name:
        The name of the temperature for which to determine the average.

    :param temperature_vector:
        The temperature vector from which the average temperature of the given layer
        should be extracted.

    :return:
        The average temperature of the layer.

    """

    layer_temperatures: Set[float] = {
        value
        for value_index, value in enumerate(temperature_vector)
        if index_handler.temperature_name_from_index(
            value_index, number_of_pipes, number_of_x_segments, number_of_y_segments
        )
        == temperature_name
    }

    try:
        average_temperature: float = sum(layer_temperatures) / len(layer_temperatures)
    except ZeroDivisionError as e:
        raise ProgrammerJudgementFault(
            "An attempt was made to compute the average temperature of a layer where "
            "no temperatures in the temperature vector matched the given temperatrure "
            f"name: {str(e)}"
        ) from None
    return average_temperature


def _calculate_vector_difference(
    first_vector: numpy.ndarray,
    second_vector: numpy.ndarray,
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
        diff_vector = [
            first_vector[index] - second_vector[index]
            for index in range(len(first_vector))
        ]
    except ValueError as e:
        raise ProgrammerJudgementFault(
            "Atempt was made to compute the difference between two vectors of "
            f"different sizes: {str(e)}"
        ) from None

    # Square the values in this vector to avoid sign issues and return the sum.
    return sum([value ** 2 for value in diff_vector])


def _date_and_time_from_time_step(
    initial_date_and_time: datetime.datetime,
    final_date_and_time: datetime.datetime,
    time_step: numpy.float64,
) -> datetime.datetime:
    """
    Returns a :class:`datetime.datetime` instance representing the current time.

    :param initial_date_and_time:
        The initial date and time for the model run.

    :param final_date_and_time:
        The final date and time for the model run.

    :param time_step:
        The current time step, measured in seconds from the start of the run.

    :return:
        The current date and time, based on the iterative point.

    :raises: ProgrammerJudgementFault
        Raised if the date and time being returned is greater than the maximum for the
        run.

    """

    date_and_time = initial_date_and_time + relativedelta(seconds=time_step)

    if date_and_time > final_date_and_time:
        raise ProgrammerJudgementFault(
            "The model reached a time step greater than the maximum time step."
        )

    return date_and_time


def _get_load_system(location: str) -> load.LoadSystem:
    """
    Instantiates a :class:`load.LoadSystem` instance based on the file data.

    :param locataion:
        The location being considered for which to instantiate the load system.

    :return:
        An instantiated :class:`load.LoadSystem` based on the input location.

    """

    load_profiles = {
        os.path.join(location, "load_profiles", filename)
        for filename in os.listdir(os.path.join(location, "load_profiles"))
    }
    load_system: load.LoadSystem = load.LoadSystem.from_data(load_profiles)
    return load_system


def _get_weather_forecaster(
    average_irradiance: bool,
    location: str,
    use_pvgis: bool,
    override_ambient_temperature: float,
    override_irradiance: float,
) -> weather.WeatherForecaster:
    """
    Instantiates a :class:`weather.WeatherForecaster` instance based on the file data.

    :param average_irradiance:
        Whether to use an average irradiance profile for the month (True) or use
        irradiance profiles for each day individually (False).

    :param location:
        The location currently being considered, and for which to instantiate the
        :class:`weather.WeatherForecaster` instance.

    :param override_ambient_temperature:
        Overrides the ambient temperature value. The value should be in degrees Celcius.

    :param override_irradiance:
        Overrides the irradiance value. The value should be in Watts per meter squared.

    :param use_pvgis:
        Whether data from the PVGIS (Photovoltaic Geographic Information Survey) should
        be used (True) or not (False).

    :return:
        An instantiated :class:`weather.WeatherForecaster` instance based on the
        supplied parameters.

    """

    solar_irradiance_filenames = {
        os.path.join(location, SOLAR_IRRADIANCE_FOLDERNAME, filename)
        for filename in os.listdir(os.path.join(location, SOLAR_IRRADIANCE_FOLDERNAME))
    }
    temperature_filenames = {
        os.path.join(location, TEMPERATURE_FOLDERNAME, filename)
        for filename in os.listdir(os.path.join(location, TEMPERATURE_FOLDERNAME))
    }

    weather_forecaster: weather.WeatherForecaster = weather.WeatherForecaster.from_data(
        average_irradiance,
        solar_irradiance_filenames,
        temperature_filenames,
        os.path.join(location, WEATHER_DATA_FILENAME),
        override_ambient_temperature,
        override_irradiance,
        use_pvgis,
    )
    return weather_forecaster


def _layer_temperature_profile(
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    segments: Dict[SegmentCoordinates, Segment],
    temperature_name: TemperatureName,
    temperature_vector: numpy.ndarray,
) -> Dict[str, float]:
    """
    Returns a map between coordinates and temperature in celcius for a layer.

    :param number_of_pipes:
        The number of pipes in the absorber.

    :param number_of_x_segments:
        The number of segments in a single row of the absorber.

    :param number_of_y_segments:
        The number of y segments in a single row of the absorber.

    :param segments:
        The list of segments involved in the layer.

    :param temperature_name:
        The name of the temperature for which to determine the average.

    :param temperature_vector:
        The temperature vector from which the average temperature of the given layer
        should be extracted.

    :return:
        A mapping between coordinates, expressed as a string, and the temperature of
        each segment, for the layer specified. The values of the temperatures are
        expressed in Celcius.

    """

    if temperature_name in [
        TemperatureName.glass,
        TemperatureName.pv,
        TemperatureName.absorber,
    ]:
        layer_temperature_map: Dict[str, float] = {
            str(segment_coordinates): temperature_vector[
                index_handler.index_from_segment_coordinates(
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    segment.x_index,
                    segment.y_index,
                )
            ]
            - ZERO_CELCIUS_OFFSET
            for segment_coordinates, segment in segments.items()
        }
    elif temperature_name in [
        TemperatureName.pipe,
        TemperatureName.htf,
        TemperatureName.htf_in,
        TemperatureName.htf_out,
    ]:
        layer_temperature_map = {
            str(segment_coordinates): temperature_vector[
                index_handler.index_from_pipe_coordinates(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    temperature_name,
                    segment.pipe_index,
                    segment.y_index,
                )
            ]
            - ZERO_CELCIUS_OFFSET
            for segment_coordinates, segment in segments.items()
            if segment.pipe
        }
    else:
        raise ProgrammerJudgementFault(
            "Attempt made to calculate a 2D profile for a temperature with no 2D "
            "nature."
        )

    return layer_temperature_map


def _solve_temperature_vector_convergence_method(
    *,
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    pvt_panel: pvt.PVT,
    run_one_temperature_vector: numpy.ndarray,
    weather_conditions: WeatherConditions,
    convergence_run_number: int = 0,
    run_one_temperature_difference: float = 5 * ZERO_CELCIUS_OFFSET ** 2,
    collector_input_temperature: Optional[float] = None,
    current_hot_water_load: Optional[float] = None,
    heat_exchanger: Optional[exchanger.Exchanger] = None,
    hot_water_tank: Optional[tank.Tank] = None,
    next_date_and_time: Optional[datetime.datetime] = None,
    previous_run_temperature_vector: Optional[numpy.ndarray] = None,
    resolution: Optional[int] = None,
) -> numpy.ndarray:
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

    :param number_of_x_segments:
        The number of segments in one "row" of the panel, i.e., which have the same y
        coordinate

    :param number_of_y_segments:
        The number of segments in one "column" of the panel, i.e., which have the same x
        coordiante.

    :param operating_mode:
        The operating mode for the run.

    :param previous_run_temperature_vector:
        The temperatures at the previous time step.

    :param pvt_panel:
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
            number_of_x_segments=number_of_x_segments,
            number_of_y_segments=number_of_y_segments,
            operating_mode=operating_mode,
            previous_temperature_vector=previous_run_temperature_vector,
            pvt_panel=pvt_panel,
            resolution=resolution,
            weather_conditions=weather_conditions,
        )
    elif operating_mode.decoupled:
        coefficient_matrix, resultant_vector = matrix.calculate_matrix_equation(
            best_guess_temperature_vector=run_one_temperature_vector,
            collector_input_temperature=collector_input_temperature,
            logger=logger,
            number_of_pipes=number_of_pipes,
            number_of_temperatures=number_of_temperatures,
            number_of_x_segments=number_of_x_segments,
            number_of_y_segments=number_of_y_segments,
            operating_mode=operating_mode,
            pvt_panel=pvt_panel,
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
    run_two_temperature_vector: numpy.ndarray = numpy.asarray(  # type: ignore
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

    if convergence_run_number == WARN_RECURSION_DEPTH:
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
    return _solve_temperature_vector_convergence_method(
        collector_input_temperature=collector_input_temperature,
        current_hot_water_load=current_hot_water_load,
        heat_exchanger=heat_exchanger,
        hot_water_tank=hot_water_tank,
        logger=logger,
        next_date_and_time=next_date_and_time,
        number_of_pipes=number_of_pipes,
        number_of_temperatures=number_of_temperatures,
        number_of_x_segments=number_of_x_segments,
        number_of_y_segments=number_of_y_segments,
        operating_mode=operating_mode,
        previous_run_temperature_vector=previous_run_temperature_vector,
        pvt_panel=pvt_panel,
        resolution=resolution,
        run_one_temperature_vector=run_two_temperature_vector,
        weather_conditions=weather_conditions,
        convergence_run_number=convergence_run_number + 1,
        run_one_temperature_difference=run_two_temperature_difference,
    )


def _system_data_from_run(
    date: datetime.date,
    initial_date_and_time: datetime.datetime,
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    pvt_panel: pvt.PVT,
    save_2d_output: bool,
    temperature_vector: numpy.ndarray,
    time: datetime.time,
    weather_conditions: WeatherConditions,
) -> SystemData:
    """
    Return a :class:`SystemData` instance based on the current system.

    :param date:
        The date relevant to the system data being saved.

    :param initial_date_and_time:
        The initial date and time for the run.

    :param number_of_pies:
        The number of pipes attached to the absorber.

    :param number_of_x_segments:
        The number of x segments included in the absorber.

    :param number_of_y_segments:
        The number of y segments included in the absorber.

    :param operating_mode:
        The operating mode for the run.

    :param pvt_panel:
        A :class:`pvt.PVT` instance representing the PVT absorber being modelled.

    :param save_2d_output:
        Whether the 2D output should be saved (True) or not (False).

    :param temperature_vector:
        The temperature vector for which to compute the system variables.

    :param time:
        The time relevant to the system data being saved.

    :param weather_conditions:
        The weather conditions at the time being saved.

    :return:
        A :class:`..__utils__.SystemData` instance containing the information about the
        system at this point during the run.

    """

    # Determine the average temperatures of the various PVT layers.
    average_glass_temperature = _average_layer_temperature(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        TemperatureName.glass,
        temperature_vector,
    )
    temperature_map_glass_layer = _layer_temperature_profile(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        pvt_panel.segments,
        TemperatureName.glass,
        temperature_vector,
    )

    average_pv_temperature = _average_layer_temperature(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        TemperatureName.pv,
        temperature_vector,
    )
    temperature_map_pv_layer = _layer_temperature_profile(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        pvt_panel.segments,
        TemperatureName.pv,
        temperature_vector,
    )

    average_absorber_temperature = _average_layer_temperature(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        TemperatureName.absorber,
        temperature_vector,
    )
    temperature_map_absorber_layer = _layer_temperature_profile(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        pvt_panel.segments,
        TemperatureName.absorber,
        temperature_vector,
    )

    average_pipe_temperature = _average_layer_temperature(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        TemperatureName.pipe,
        temperature_vector,
    )
    temperature_map_pipe_layer = _layer_temperature_profile(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        pvt_panel.segments,
        TemperatureName.pipe,
        temperature_vector,
    )

    average_bulk_water_temperature = _average_layer_temperature(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        TemperatureName.htf,
        temperature_vector,
    )
    temperature_map_bulk_water_layer = _layer_temperature_profile(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        pvt_panel.segments,
        TemperatureName.htf,
        temperature_vector,
    )

    # Set various parameters depending on the operating mode of the model.
    # Set the variables that depend on a coupled vs decoupled system.
    if operating_mode.coupled:
        exchanger_temperature_drop: Optional[float] = (
            temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank_out,
                )
            ]
            - temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank_in,
                )
            ]
            if temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank_in,
                )
            ]
            > temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank,
                )
            ]
            else 0
        )
        tank_temperature: Optional[float] = (
            temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank,
                )
            ]
            - ZERO_CELCIUS_OFFSET
        )
    else:
        exchanger_temperature_drop = None
        tank_temperature = None
    # Set the variables that depend on a dynamic vs steady-state system.
    if operating_mode.dynamic:
        time = (
            str((date.day - initial_date_and_time.day) * 24 + time.hour)
            + time.strftime("%H:%M:%S")[2:]
        )
    else:
        time = None

    # Compute variables in common to both.
    collector_input_temperature = (
        temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector_in,
            )
        ]
        - ZERO_CELCIUS_OFFSET
    )
    collector_output_temperature = (
        temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector_out,
            )
        ]
        - ZERO_CELCIUS_OFFSET
    )

    # Return the system data.
    return SystemData(
        date=date.strftime("%d/%m/%Y"),
        time=time,
        glass_temperature=average_glass_temperature - ZERO_CELCIUS_OFFSET,
        pv_temperature=average_pv_temperature - ZERO_CELCIUS_OFFSET,
        absorber_temperature=average_absorber_temperature - ZERO_CELCIUS_OFFSET,
        collector_input_temperature=collector_input_temperature,
        collector_output_temperature=collector_output_temperature,
        pipe_temperature=average_pipe_temperature - ZERO_CELCIUS_OFFSET,
        bulk_water_temperature=average_bulk_water_temperature - ZERO_CELCIUS_OFFSET,
        ambient_temperature=weather_conditions.ambient_temperature
        - ZERO_CELCIUS_OFFSET,
        exchanger_temperature_drop=exchanger_temperature_drop,
        tank_temperature=tank_temperature,
        sky_temperature=weather_conditions.sky_temperature - ZERO_CELCIUS_OFFSET,
        layer_temperature_map_bulk_water=temperature_map_bulk_water_layer
        if save_2d_output
        else None,
        layer_temperature_map_absorber=temperature_map_absorber_layer
        if save_2d_output
        else None,
        layer_temperature_map_glass=temperature_map_glass_layer
        if save_2d_output
        else None,
        layer_temperature_map_pipe=temperature_map_pipe_layer
        if save_2d_output
        else None,
        layer_temperature_map_pv=temperature_map_pv_layer if save_2d_output else None,
        thermal_efficiency=efficiency.thermal_efficiency(
            pvt_panel.area,
            pvt_panel.absorber.mass_flow_rate,
            weather_conditions.irradiance,
            collector_output_temperature - collector_input_temperature,
        ),
        reduced_temperature=300,
    )


######################
# System run methods #
######################


def _dynamic_system_run(
    cloud_efficacy_factor: float,
    days: int,
    heat_exchanger: exchanger.Exchanger,
    hot_water_tank: tank.Tank,
    initial_month: int,
    initial_system_temperature_vector: List[float],
    load_system: load.LoadSystem,
    logger: logging.Logger,
    months: int,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    pvt_panel: pvt.PVT,
    resolution: int,
    save_2d_output: bool,
    start_time: int,
    weather_forecaster: weather.WeatherForecaster,
) -> Tuple[numpy.ndarray, Dict[int, SystemData]]:
    """
    Carries out a dynamic run of the system.

    :param cloud_efficacy_factor:
        How effective the cloud cover is, from 0 (no effect) to 1 (no solar irradiance
        makes it through the cloud layer).

    :param days:
        How many days the system should run the simulation for.

    :param heat_exchanger:
        The heat exchanger between the HTF and the hot-water tank.

    :param hot_water_tank:
        A :class:`tank.Tank` instance representing the hot-water tank being included in
        the system.

    :param initial_month:
        The month for which the simulation should start to be run.

    :param initial_system_temperature_vector:
        The set of system temperatures, expressed as a `list`, at the beginning of the
        run.

    :param load_system:
        The load system for the run.

    :param logger:
        The logger to be used for the run.

    :param months:
        The number of months for which to run the simulation, if specified. Otherwise,
        `None`.

    :param number_of_pipes:
        The number of pipes attached to the absorber.

    :param number_of_temperatures:
        The number of temperatures being modelled.

    :param number_of_x_segments:
        The number of segments in the x direction.

    :param number_of_y_segments:
        The number of segments in the y direction.

    :param operating_mode:
        The operating mode of the system.

    :param pvt_panel:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The temporal resolution (in seconds) for the run.

    :param save_2d_output:
        Whether the 2D output should be saved (True) or not (False).

    :param start_time:
        The start time, expressed in hours from midnight, at which to begin running the
        model.

    :param weather_forecaster:
        The weather forecaster for the system.

    :return:
        A `tuple` containing:
        - the final system-state reached by the model,
        - the system data for the run.

    """

    # Set up a holder for information about the system.
    system_data: Dict[int, SystemData] = dict()

    # Set up the time iterator.
    num_months = (
        (initial_month if initial_month is not None else 1) - 1 + months
    )  # [months]
    start_month = (
        initial_month
        if 1 <= initial_month <= 12
        else DEFAULT_INITIAL_DATE_AND_TIME.month
    )  # [months]
    initial_date_and_time = DEFAULT_INITIAL_DATE_AND_TIME.replace(
        hour=start_time, month=start_month
    )
    if days is None:
        final_date_and_time = initial_date_and_time + relativedelta(
            months=num_months % 12, years=num_months // 12
        )
    else:
        final_date_and_time = initial_date_and_time + relativedelta(days=days)

    logger.info(
        "Beginning itterative model:\n  Running from: %s\n  Running to: %s",
        str(initial_date_and_time),
        str(final_date_and_time),
    )

    time_iterator_step = relativedelta(seconds=resolution)

    weather_conditions = weather_forecaster.get_weather(
        pvt_panel.latitude,
        pvt_panel.longitude,
        cloud_efficacy_factor,
        initial_date_and_time,
    )

    system_data[0] = _system_data_from_run(
        initial_date_and_time.date(),
        initial_date_and_time,
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        operating_mode,
        pvt_panel,
        save_2d_output,
        initial_system_temperature_vector,
        initial_date_and_time.time(),
        weather_conditions,
    )

    previous_run_temperature_vector = initial_system_temperature_vector

    for run_number, date_and_time in enumerate(
        time_iterator(
            first_time=initial_date_and_time,
            last_time=final_date_and_time,
            resolution=resolution,
            timezone=pvt_panel.timezone,
        )
    ):

        logger.debug(
            "Time: %s: Beginning internal run. Previous temperature vector: T=%s",
            date_and_time.strftime("%d/%m/%Y %H:%M:%S"),
            previous_run_temperature_vector,
        )

        # Determine the "i+1" time.
        next_date_and_time = date_and_time + time_iterator_step

        # Determine the "i+1" current hot-water load.
        current_hot_water_load = (
            load_system[
                (load.ProfileType.HOT_WATER, next_date_and_time)
            ]  # [litres/hour]
            / 3600  # [seconds/hour]
        )  # [kg/s]

        # Determine the "i+1" current weather conditions.
        weather_conditions = weather_forecaster.get_weather(
            pvt_panel.latitude,
            pvt_panel.longitude,
            cloud_efficacy_factor,
            next_date_and_time,
        )

        try:
            current_run_temperature_vector = (
                _solve_temperature_vector_convergence_method(
                    current_hot_water_load=current_hot_water_load,
                    heat_exchanger=heat_exchanger,
                    hot_water_tank=hot_water_tank,
                    logger=logger,
                    next_date_and_time=next_date_and_time,
                    number_of_pipes=number_of_pipes,
                    number_of_temperatures=number_of_temperatures,
                    number_of_x_segments=number_of_x_segments,
                    number_of_y_segments=number_of_y_segments,
                    operating_mode=operating_mode,
                    previous_run_temperature_vector=previous_run_temperature_vector,
                    pvt_panel=pvt_panel,
                    resolution=resolution,
                    run_one_temperature_vector=previous_run_temperature_vector,
                    weather_conditions=weather_conditions,
                )
            )
        except DivergentSolutionError as e:
            logger.error(
                "A divergent solution was reached at %s:%s",
                date_and_time.strftime("%D/%M/%Y %H:%M:%S"),
                str(e),
            )
            raise

        # Save the system data output and 2D profiles.
        system_data[run_number + 1] = _system_data_from_run(
            date_and_time.date(),
            initial_date_and_time,
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            operating_mode,
            pvt_panel,
            save_2d_output,
            current_run_temperature_vector,
            date_and_time.time(),
            weather_conditions,
        )
        previous_run_temperature_vector = current_run_temperature_vector

    return current_run_temperature_vector, system_data


def _steady_state_run(
    collector_input_temperature: float,
    cloud_efficacy_factor: float,
    initial_date_and_time: datetime.datetime,
    initial_system_temperature_vector: List[float],
    logger: logging.Logger,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    pvt_panel: pvt.PVT,
    save_2d_output: bool,
    weather_forecaster: weather.WeatherForecaster,
) -> Tuple[numpy.ndarray, Dict[int, SystemData]]:
    """
    Carries out a steady-state run of the system.

    """

    # Set up a holder for information about the system.
    system_data: Dict[int, SystemData] = dict()

    # Set up various variables needed to model the system.
    weather_conditions = weather_forecaster.get_weather(
        pvt_panel.latitude,
        pvt_panel.longitude,
        cloud_efficacy_factor,
    )

    logger.info(
        "Beginning steady-state model:\n  Running on: %s\n  Running at convergent "
        "precision: %sK",
        initial_date_and_time.date().strftime("%d/%m/%Y"),
        str(10 ** -CONVERGENT_SOLUTION_PRECISION),
    )

    system_data[0] = _system_data_from_run(
        initial_date_and_time.date(),
        initial_date_and_time,
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        operating_mode,
        pvt_panel,
        save_2d_output,
        initial_system_temperature_vector,
        initial_date_and_time.time(),
        weather_conditions,
    )

    try:
        current_run_temperature_vector = _solve_temperature_vector_convergence_method(
            collector_input_temperature=collector_input_temperature,
            logger=logger,
            number_of_pipes=number_of_pipes,
            number_of_temperatures=number_of_temperatures,
            number_of_x_segments=number_of_x_segments,
            number_of_y_segments=number_of_y_segments,
            operating_mode=operating_mode,
            pvt_panel=pvt_panel,
            run_one_temperature_vector=initial_system_temperature_vector,
            weather_conditions=weather_conditions,
        )
    except DivergentSolutionError as e:
        logger.error(
            "A divergent solution was when attempting to solve the system in steady-state:%s",
            str(e),
        )
        raise

    # Save the system data output and 2D profiles.
    system_data[1] = _system_data_from_run(
        initial_date_and_time.date(),
        initial_date_and_time,
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        operating_mode,
        pvt_panel,
        save_2d_output,
        current_run_temperature_vector,
        initial_date_and_time.time(),
        weather_conditions,
    )

    return current_run_temperature_vector, system_data


def main(
    average_irradiance: bool,
    cloud_efficacy_factor: float,
    exchanger_data_file: str,
    initial_month: int,
    initial_system_temperature_vector: List[float],
    layers: Set[TemperatureName],
    location: str,
    operating_mode: OperatingMode,
    portion_covered: float,
    pvt_data_file: str,
    resolution: int,
    save_2d_output: bool,
    tank_data_file: str,
    use_pvgis: bool,
    verbose: bool,
    x_resolution: int,
    y_resolution: int,
    *,
    override_ambient_temperature: Optional[float],
    override_irradiance: Optional[float],
    run_number: Optional[int],
    start_time: Optional[int],
    days: Optional[int] = None,
    months: Optional[int] = None,
) -> Tuple[numpy.ndarray, Dict[int, SystemData]]:
    """
    The main module for the code. Calling this method executes a run of the simulation.

    :param average_irradiance:
        Whether to use an average irradiance profile for the month.

    :param cloud_efficiacy_factor:
        The extent to which cloud cover influences the solar irradiance.

    :param exchanger_data_file:
        The path to the data file containing information about the heat exchanger being
        modelled in the run.

    :param initial_month:
        The initial month for the run, given between 1 (January) and 12 (December).

    :param initial_system_temperature_vector:
        The vector of initial temperatures used to model the system.

    :param location:
        The location at which the PVT system is installed and for which location data
        should be used.

    :param operating_mode:
        The operating mode of the run, containing information needed to set up the
        matrix.

    :param portion_covered:
        The portion of the absorber which is covered with PV cells.

    :param pvt_data_file:
        The path to the data file containing information about the PVT system being
        modelled.

    :param resolution:
        The temporal resolution at which to run the simulation.

    :param save_2d_output:
        If True, the 2D output is saved to the system data and returned. If False, only
        the 1D output is saved.

    :param tank_data_file:
        The path to the data file containing information about the hot-water tank used
        in the model.

    :param unglazed:
        If specified, the glass cover is not included in the model.

    :param use_pvgis:
        Whether the data obtained from the PVGIS system should be used.

    :param verbose:
        Whether the logging level is verbose (True) or not (False).

    :param x_resolution:
        The x resolution of the simulation being run.

    :param y_resolution:
        The y resolution of the simulation being run.

    :param override_ambient_temperature:
        In decoupled instances, the ambient temperature can be specified as a constant
        value which will override the ambient-temperature profiles.

    :param override_irradiance:
        In decoupled instances, the solar irradiance can be specified as a constant
        value which will override the solar-irradiance profiles.

    :param run_number:
        The number of the run being carried out. This is used for categorising logs.

    :param start_time:
        The time of day at which to start the simulation, specified between 0 and 23.
        This can be `None` if a steady-state simulation is being run.

    :param days:
        The number of days for which the simulation is being run. This can be `None` if
        a steady-state simulation is being run.

    :param months:
        The number of months for which to run the simulation. This can be `None` if a
        steady-state simulation is being run.

    :return:
        The system data is returned.

    """

    # Get the logger for the component.
    if operating_mode.dynamic:
        logger = get_logger(
            PVT_SYSTEM_MODEL_LOGGER_NAME.format(
                tag=f"{resolution}s", run_number=run_number
            ),
            verbose,
        )
    else:
        logger = get_logger(
            PVT_SYSTEM_MODEL_LOGGER_NAME.format(tag="dynamic", run_number=run_number),
            verbose,
        )

    # Set up numpy printing style.
    numpy.set_printoptions(formatter={"float": "{: 0.3f}".format})

    # Set up the weather module.
    weather_forecaster = _get_weather_forecaster(
        average_irradiance,
        location,
        use_pvgis,
        override_ambient_temperature,
        override_irradiance,
    )
    logger.info("Weather forecaster successfully instantiated: %s", weather_forecaster)

    # Set up the load module.
    load_system = _get_load_system(location)
    logger.info(
        "Load system successfully instantiated: %s",
        load_system,
    )

    # Raise an exception if no initial system temperature vector was supplied.
    if initial_system_temperature_vector is None:
        raise ProgrammerJudgementFault(
            "Not initial system temperature vector was supplied. This is necessary "
            "when calling the pvt model."
        )

    # Initialise the PV-T panel.
    pvt_panel = process_pvt_system_data.pvt_panel_from_path(
        layers,
        logger,
        portion_covered,
        pvt_data_file,
        x_resolution,
        y_resolution,
    )
    logger.info("PV-T panel successfully instantiated: %s", pvt_panel)
    logger.info(
        "PV-T panel segments:\n  %s",
        "\n  ".join(
            [
                f"{segment_coordinates}: {segment}"
                for segment_coordinates, segment in pvt_panel.segments.items()
            ]
        ),
    )

    # Instantiate the rest of the PVT system if relevant.
    if operating_mode.coupled:
        # Set up the heat exchanger.
        heat_exchanger: Optional[
            exchanger.Exchanger
        ] = process_pvt_system_data.heat_exchanger_from_path(exchanger_data_file)
        logger.info("Heat exchanger successfully instantiated: %s", heat_exchanger)

        # Set up the hot-water tank.
        hot_water_tank: Optional[
            tank.Tank
        ] = process_pvt_system_data.hot_water_tank_from_path(tank_data_file)
        logger.info("Hot-water tank successfully instantiated: %s", hot_water_tank)

        # Set up the mains supply system.
        mains_supply = mains_power.MainsSupply.from_yaml(
            os.path.join(location, "utilities.yaml")
        )
        logger.info("Mains supply successfully instantiated: %s", mains_supply)

    else:
        heat_exchanger = None
        logger.info("NO HEAT EXCHANGER PRESENT")
        hot_water_tank = None
        logger.info("NO HOT-WATER TANK PRESENT")
        mains_supply = None
        logger.info("NO MAINS SUPPLY PRESENT")

    # Determine the number of temperatures being modelled.
    number_of_pipes = len(
        {
            segment.pipe_index
            for segment in pvt_panel.segments.values()
            if segment.pipe_index is not None
        }
    )
    number_of_x_segments = len(
        {segment.x_index for segment in pvt_panel.segments.values()}
    )
    number_of_y_segments = len(
        {segment.y_index for segment in pvt_panel.segments.values()}
    )
    if operating_mode.coupled:
        number_of_temperatures: int = index_handler.num_temperatures(
            number_of_pipes, number_of_x_segments, number_of_y_segments
        )
    else:
        number_of_temperatures: int = (
            index_handler.num_temperatures(
                number_of_pipes, number_of_x_segments, number_of_y_segments
            )
            - 3
        )
    logger.info(
        "System consists of %s pipes, %s by %s segments, and %s temperatures in all.",
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        number_of_temperatures,
    )

    # Instantiate the two pipes used to store input and output temperature values.
    # absorber_to_tank_pipe = pipe.Pipe(temperature=initial_system_temperature_vector[3])
    # tank_to_absorber_pipe = pipe.Pipe(temperature=initial_system_temperature_vector[3])

    # Instnatiate the hot-water pump.
    # htf_pump = process_pvt_system_data.pump_from_path(pump_data_file)

    # Set up a holder for the information about the final output of the system.
    # total_power_data = TotalPowerData()

    logger.info(
        "System state before beginning run:\n%s\n%s\n%s\n%s",
        heat_exchanger if heat_exchanger is not None else "No heat exchanger",
        hot_water_tank if hot_water_tank is not None else "No hot-water tank",
        pvt_panel,
        weather_forecaster,
    )

    if operating_mode.dynamic:
        final_run_temperature_vector, system_data = _dynamic_system_run(
            cloud_efficacy_factor,
            days,
            heat_exchanger,
            hot_water_tank,
            initial_month,
            initial_system_temperature_vector,
            load_system,
            logger,
            months,
            number_of_pipes,
            number_of_temperatures,
            number_of_x_segments,
            number_of_y_segments,
            operating_mode,
            pvt_panel,
            resolution,
            save_2d_output,
            start_time,
            weather_forecaster,
        )
    elif operating_mode.steady_state:
        system_data = {
            collector_input_temperature: _steady_state_run(
                collector_input_temperature,
                cloud_efficacy_factor,
                DEFAULT_INITIAL_DATE_AND_TIME.replace(month=initial_month),
                initial_system_temperature_vector,
                logger,
                number_of_pipes,
                number_of_temperatures,
                number_of_x_segments,
                number_of_y_segments,
                operating_mode,
                pvt_panel,
                save_2d_output,
                weather_forecaster,
            )[1][1]
            for collector_input_temperature in COLLECTOR_INPUT_TEMPERATURES
        }
        final_run_temperature_vector = None
    else:
        raise ProgrammerJudgementFault(
            "The system model was called with an operating mode "
        )

    return final_run_temperature_vector, system_data


if __name__ == "__main__":
    logging.error(
        "Calling the internal `pvt_system_model` from the command-line is no longer "
        "supported."
    )
    raise ProgrammerJudgementFault(
        "Calling the internal `pvt_system_model` from the command-line interface is no "
        "longer supported."
    )
