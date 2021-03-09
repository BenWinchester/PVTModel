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

from .constants import (
    CONVERGENT_SOLUTION_PRECISION,
    ZERO_CELCIUS_OFFSET,
)

from ..__utils__ import (  # pylint: disable=unused-import
    CarbonEmissions,
    get_logger,
    SystemData,
    TemperatureName,
    TotalPowerData,
)
from .__utils__ import (
    DivergentSolutionError,
    ProgrammerJudgementFault,
    PVT_SYSTEM_MODEL_LOGGER_NAME,
    time_iterator,
)


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
        The number of pipes in the collector.

    :param number_of_x_segments:
        The number of segments in a single row of the collector.

    :param number_of_y_segments:
        The number of y segments in a single row of the collector.

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
    average_irradiance: bool, location: str, use_pvgis: bool
) -> weather.WeatherForecaster:
    """
    Instantiates a :class:`weather.WeatherForecaster` instance based on the file data.

    :param average_irradiance:
        Whether to use an average irradiance profile for the month (True) or use
        irradiance profiles for each day individually (False).

    :param location:
        The location currently being considered, and for which to instantiate the
        :class:`weather.WeatherForecaster` instance.

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
        os.path.join(location, WEATHER_DATA_FILENAME),
        solar_irradiance_filenames,
        temperature_filenames,
        use_pvgis,
    )
    return weather_forecaster


def _solve_temperature_vector_convergence_method(
    current_hot_water_load: float,
    heat_exchanger: exchanger.Exchanger,
    hot_water_tank: tank.Tank,
    logger: logging.Logger,
    next_date_and_time: datetime.datetime,
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    previous_run_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
    run_one_temperature_vector: numpy.ndarray,
    weather_conditions: weather.WeatherConditions,
    convergence_run_number: int = 0,
    run_one_temperature_difference: float = 5 * ZERO_CELCIUS_OFFSET ** 2,
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
        "Date and time: %s; Run number: %s: " "Beginning convergent calculation.",
        next_date_and_time.strftime("%d/%m/%Y %H:%M:%S"),
        convergence_run_number,
    )

    coefficient_matrix, resultant_vector = matrix.calculate_matrix_equation(
        run_one_temperature_vector,
        heat_exchanger,
        current_hot_water_load,
        hot_water_tank,
        number_of_pipes,
        number_of_temperatures,
        number_of_x_segments,
        number_of_y_segments,
        previous_run_temperature_vector,
        pvt_panel,
        resolution,
        weather_conditions,
    )

    logger.debug(
        "Matrix equation computed.\nA =\n%s\nB =\n%s",
        str(coefficient_matrix),
        str(resultant_vector),
    )

    run_two_output = linalg.solve(a=coefficient_matrix, b=resultant_vector)
    # run_two_output = linalg.lstsq(a=coefficient_matrix, b=resultant_vector)
    run_two_temperature_vector: numpy.ndarray = numpy.asarray(  # type: ignore
        [run_two_output[index][0] for index in range(len(run_two_output))]
    )
    # run_two_temperature_vector = run_two_output[0].transpose()[0]

    logger.debug(
        "Date and time: %s; Run number: %s: "
        "Temperatures successfully computed. Temperature vector: T = %s",
        next_date_and_time.strftime("%d/%m/%Y %H:%M:%S"),
        convergence_run_number,
        run_two_temperature_vector,
    )

    run_two_temperature_difference = _calculate_vector_difference(
        run_one_temperature_vector, run_two_temperature_vector
    )

    # If the solution has converged, return the temperature vector.
    if run_two_temperature_difference < CONVERGENT_SOLUTION_PRECISION:
        logger.debug(
            "Date and time: %s; Run number: %s: Convergent solution found. "
            "Convergent difference: %s",
            next_date_and_time.strftime("%d/%m/%Y %H:%M:%S"),
            convergence_run_number,
            run_two_temperature_difference,
        )
        return run_two_temperature_vector

    # If the solution has diverged, raise an Exception.
    if run_two_temperature_difference > run_one_temperature_difference:
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
        current_hot_water_load=current_hot_water_load,
        heat_exchanger=heat_exchanger,
        hot_water_tank=hot_water_tank,
        logger=logger,
        next_date_and_time=next_date_and_time,
        number_of_pipes=number_of_pipes,
        number_of_temperatures=number_of_temperatures,
        number_of_x_segments=number_of_x_segments,
        number_of_y_segments=number_of_y_segments,
        previous_run_temperature_vector=previous_run_temperature_vector,
        pvt_panel=pvt_panel,
        resolution=resolution,
        run_one_temperature_vector=run_two_temperature_vector,
        weather_conditions=weather_conditions,
        convergence_run_number=convergence_run_number + 1,
        run_one_temperature_difference=run_two_temperature_difference,
    )


def main(
    average_irradiance: bool,
    cloud_efficacy_factor: float,
    days: int,
    exchanger_data_file: str,
    initial_month: int,
    initial_system_temperature_vector: List[float],
    location: str,
    months: int,
    portion_covered: float,
    pvt_data_file: str,
    resolution: int,
    run_number: Optional[int],
    start_time: int,
    tank_data_file: str,
    use_pvgis: bool,
    verbose: bool,
    x_resolution: int,
    y_resolution: int,
) -> Tuple[numpy.ndarray, Dict[int, SystemData]]:
    """
    The main module for the code. Calling this method executes a run of the simulation.

    :param average_irradiance:
        Whether to use an average irradiance profile for the month.

    :param cloud_efficiacy_factor:
        The extent to which cloud cover influences the solar irradiance.

    :param days:
        The number of days for which the simulation is being run.

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

    :param months:
        The number of months for which to run the simulation.

    :param portion_covered:
        The portion of the collector which is covered with PV cells.

    :param pvt_data_file:
        The path to the data file containing information about the PVT system being
        modelled.

    :param resolution:
        The temporal resolution at which to run the simulation.

    :param run_number:
        The number of the run being carried out. This is used for categorising logs.

    :param start_time:
        The time of day at which to start the simulation, specified between 0 and 23.

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

    :return:
        The system data is returned.

    """

    # Get the logger for the component.
    logger = get_logger(
        PVT_SYSTEM_MODEL_LOGGER_NAME.format(
            resolution=resolution, run_number=run_number
        ),
        verbose
    )

    # Set up numpy printing style.
    numpy.set_printoptions(formatter={"float": "{: 0.3f}".format})

    # Set up the weather module.
    weather_forecaster = _get_weather_forecaster(
        average_irradiance, location, use_pvgis
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
        portion_covered,
        pvt_data_file,
        x_resolution,
        y_resolution,
    )
    logger.info("PV-T panel successfully instantiated: %s", pvt_panel)

    # Instantiate the rest of the PVT system.
    heat_exchanger = process_pvt_system_data.heat_exchanger_from_path(
        exchanger_data_file
    )
    logger.info("Heat exchanger successfully instantiated: %s", heat_exchanger)
    hot_water_tank = process_pvt_system_data.hot_water_tank_from_path(tank_data_file)
    logger.info("Hot-water tank successfully instantiated: %s", hot_water_tank)

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
    number_of_temperatures = index_handler.num_temperatures(
        number_of_pipes, number_of_x_segments, number_of_y_segments
    )
    logger.info(
        "System consists of %s pipes, %s by %s segments, and %s temperatures in all.",
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        number_of_temperatures,
    )

    # Instantiate the two pipes used to store input and output temperature values.
    # collector_to_tank_pipe = pipe.Pipe(temperature=initial_system_temperature_vector[3])
    # tank_to_collector_pipe = pipe.Pipe(temperature=initial_system_temperature_vector[3])

    # Instnatiate the hot-water pump.
    # htf_pump = process_pvt_system_data.pump_from_path(pump_data_file)

    # Intiailise the mains supply system.
    mains_supply = mains_power.MainsSupply.from_yaml(
        os.path.join(location, "utilities.yaml")
    )
    logger.info("Mains supply successfully instantiated: %s", mains_supply)

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

    # Set up a holder for information about the system.
    system_data: Dict[int, SystemData] = dict()

    # Set up a holder for the information about the final output of the system.
    # total_power_data = TotalPowerData()

    logger.info(
        "Beginning itterative model:\n  Running from: %s\n  Running to: %s",
        str(initial_date_and_time),
        str(final_date_and_time),
    )

    logger.info(
        "System state before beginning run:\n%s\n%s\n%s\n%s",
        heat_exchanger,
        hot_water_tank,
        pvt_panel,
        weather_forecaster,
    )

    previous_run_temperature_vector: numpy.ndarray = numpy.asarray(  # type: ignore
        initial_system_temperature_vector
    )
    time_iterator_step = relativedelta(seconds=resolution)

    # Save the initial system data.
    weather_conditions = weather_forecaster.get_weather(
        pvt_panel.latitude,
        pvt_panel.longitude,
        cloud_efficacy_factor,
        initial_date_and_time,
    )

    # Determine the average temperatures for the layers in the PV-T system.
    average_glass_temperature = _average_layer_temperature(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        TemperatureName.glass,
        previous_run_temperature_vector,
    )
    average_pv_temperature = _average_layer_temperature(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        TemperatureName.pv,
        previous_run_temperature_vector,
    )
    average_collector_temperature = _average_layer_temperature(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        TemperatureName.collector,
        previous_run_temperature_vector,
    )
    average_bulk_water_temperature = _average_layer_temperature(
        number_of_pipes,
        number_of_x_segments,
        number_of_y_segments,
        TemperatureName.htf,
        previous_run_temperature_vector,
    )

    # Save the system data.
    system_data[0] = SystemData(
        date=initial_date_and_time.strftime("%d/%m/%Y"),
        time=initial_date_and_time.strftime("%H:%M:%S"),
        glass_temperature=average_glass_temperature - ZERO_CELCIUS_OFFSET,
        pv_temperature=average_pv_temperature - ZERO_CELCIUS_OFFSET,
        collector_temperature=average_collector_temperature - ZERO_CELCIUS_OFFSET,
        collector_input_temperature=previous_run_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector_in,
            )
        ]
        - ZERO_CELCIUS_OFFSET,
        collector_output_temperature=previous_run_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.collector_out,
            )
        ]
        - ZERO_CELCIUS_OFFSET,
        bulk_water_temperature=average_bulk_water_temperature - ZERO_CELCIUS_OFFSET,
        ambient_temperature=weather_conditions.ambient_temperature
        - ZERO_CELCIUS_OFFSET,
        exchanger_temperature_drop=previous_run_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank_out,
            )
        ]
        - previous_run_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank_in,
            )
        ]
        if previous_run_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank_in,
            )
        ]
        > previous_run_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank,
            )
        ]
        else 0,
        tank_temperature=previous_run_temperature_vector[
            index_handler.index_from_temperature_name(
                number_of_pipes,
                number_of_x_segments,
                number_of_y_segments,
                TemperatureName.tank,
            )
        ]
        - ZERO_CELCIUS_OFFSET,
        sky_temperature=weather_conditions.sky_temperature - ZERO_CELCIUS_OFFSET,
    )

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
                    previous_run_temperature_vector=previous_run_temperature_vector,
                    pvt_panel=pvt_panel,
                    resolution=resolution,
                    run_one_temperature_vector=previous_run_temperature_vector,
                    weather_conditions=weather_conditions,
                )
            )
        except DivergentSolutionError as e:
            logger.error(
                "A divergent solution was reached at %s.",
                date_and_time.strftime("%D/%M/%Y %H:%M:%S"),
            )
            raise

        # Determine the average temperatures of the various PVT layers.
        average_glass_temperature = _average_layer_temperature(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.glass,
            current_run_temperature_vector,
        )
        average_pv_temperature = _average_layer_temperature(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.pv,
            current_run_temperature_vector,
        )
        average_collector_temperature = _average_layer_temperature(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector,
            current_run_temperature_vector,
        )
        average_bulk_water_temperature = _average_layer_temperature(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.htf,
            current_run_temperature_vector,
        )

        # Save the system data.
        system_data[run_number + 1] = SystemData(
            date=next_date_and_time.strftime("%d/%m/%Y"),
            time=str(
                (next_date_and_time.day - initial_date_and_time.day) * 24
                + next_date_and_time.hour
            )
            + next_date_and_time.strftime("%H:%M:%S")[2:],
            glass_temperature=average_glass_temperature - ZERO_CELCIUS_OFFSET,
            pv_temperature=average_pv_temperature - ZERO_CELCIUS_OFFSET,
            collector_temperature=average_collector_temperature - ZERO_CELCIUS_OFFSET,
            collector_input_temperature=current_run_temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.collector_in,
                )
            ]
            - ZERO_CELCIUS_OFFSET,
            collector_output_temperature=current_run_temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.collector_out,
                )
            ]
            - ZERO_CELCIUS_OFFSET,
            bulk_water_temperature=average_bulk_water_temperature - ZERO_CELCIUS_OFFSET,
            ambient_temperature=weather_conditions.ambient_temperature
            - ZERO_CELCIUS_OFFSET,
            exchanger_temperature_drop=current_run_temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank_out,
                )
            ]
            - current_run_temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank_in,
                )
            ]
            if current_run_temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank_in,
                )
            ]
            > current_run_temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank,
                )
            ]
            else 0,
            tank_temperature=current_run_temperature_vector[
                index_handler.index_from_temperature_name(
                    number_of_pipes,
                    number_of_x_segments,
                    number_of_y_segments,
                    TemperatureName.tank,
                )
            ]
            - ZERO_CELCIUS_OFFSET,
            sky_temperature=weather_conditions.sky_temperature - ZERO_CELCIUS_OFFSET,
        )

        previous_run_temperature_vector = current_run_temperature_vector

    return current_run_temperature_vector, system_data


if __name__ == "__main__":
    logging.error(
        "Calling the internal `pvt_system_model` from the command-line is no longer "
        "supported."
    )
    raise ProgrammerJudgementFault(
        "Calling the internal `pvt_system_model` from the command-line interface is no "
        "longer supported."
    )
