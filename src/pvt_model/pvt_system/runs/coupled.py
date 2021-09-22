#!/usr/bin/python3.7
########################################################################################
# coupled.py - Module for executing coupled system runs.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The coupled-run module for the PV-T model.

This module exposes methods needed to carry out a coupled run of the system.

A coupled run treates the PV-T collector as connected to the rest of the system (tank
etc.) with demand profiles being utilised accordingly.

"""

import logging

from typing import Dict, List, Optional, Tuple, Union

import numpy

from dateutil.relativedelta import relativedelta

from .. import convergent_solver, exchanger, load, tank, weather
from ..pvt_collector import pvt

from ...__utils__ import BColours, OperatingMode, ProgrammerJudgementFault, SystemData
from ..__utils__ import DivergentSolutionError, time_iterator
from ..constants import DEFAULT_INITIAL_DATE_AND_TIME
from .__utils__ import system_data_from_run

__all__ = ("coupled_dynamic_run",)


def coupled_dynamic_run(
    cloud_efficacy_factor: float,
    days: Optional[int],
    heat_exchanger: exchanger.Exchanger,
    hot_water_tank: tank.Tank,
    initial_month: int,
    initial_system_temperature_vector: List[float],
    load_system: load.LoadSystem,
    logger: logging.Logger,
    minutes: Optional[int],
    months: Optional[int],
    number_of_pipes: int,
    number_of_temperatures: int,
    number_of_x_elements: int,
    number_of_y_elements: int,
    operating_mode: OperatingMode,
    pvt_collector: pvt.PVT,
    resolution: int,
    save_2d_output: bool,
    start_time: int,
    weather_forecaster: weather.WeatherForecaster,
) -> Tuple[numpy.ndarray, Dict[float, SystemData]]:
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

    :param minutes:
        The number of minutes for which to run the simulation, if specified. Otherwise,
        `None`.

    :param months:
        The number of months for which to run the simulation, if specified. Otherwise,
        `None`.

    :param number_of_pipes:
        The number of pipes attached to the absorber.

    :param number_of_temperatures:
        The number of temperatures being modelled.

    :param number_of_x_elements:
        The number of elements in the x direction.

    :param number_of_y_elements:
        The number of elements in the y direction.

    :param operating_mode:
        The operating mode of the system.

    :param pvt_collector:
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
    system_data: Dict[float, SystemData] = dict()

    # Set up the time iterator.
    if minutes is None and days is None and months is None:
        raise ProgrammerJudgementFault(
            "{}Either minutes, days or months must be specified for dynamic ".format(
                BColours.FAIL
            )
            + "runs.{}".format(BColours.ENDC)
        )

    num_months = (
        (initial_month if initial_month is not None else 1)
        - 1
        + (months if months is not None else 0)
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
        pvt_collector.latitude,
        pvt_collector.longitude,
        cloud_efficacy_factor,
        initial_date_and_time,
    )

    system_data[0] = system_data_from_run(
        initial_date_and_time.date(),
        initial_date_and_time,
        number_of_pipes,
        number_of_x_elements,
        operating_mode,
        pvt_collector,
        save_2d_output,
        initial_system_temperature_vector,
        initial_date_and_time.time(),
        weather_conditions,
    )

    previous_run_temperature_vector: Union[
        List[float], numpy.ndarray
    ] = initial_system_temperature_vector

    for run_number, date_and_time in enumerate(
        time_iterator(
            first_time=initial_date_and_time,
            last_time=final_date_and_time,
            resolution=resolution,
            timezone=pvt_collector.timezone,
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
        try:
            weather_conditions = weather_forecaster.get_weather(
                pvt_collector.latitude,
                pvt_collector.longitude,
                cloud_efficacy_factor,
                next_date_and_time,
            )
        except KeyError as e:
            logger.error("Failed to get weather conditions, do all days have profiles?")
            raise

        try:
            current_run_temperature_vector = (
                convergent_solver.solve_temperature_vector_convergence_method(
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
        system_data[run_number + 1] = system_data_from_run(
            next_date_and_time.date(),
            initial_date_and_time,
            number_of_pipes,
            number_of_x_elements,
            operating_mode,
            pvt_collector,
            save_2d_output,
            current_run_temperature_vector,
            next_date_and_time.time(),
            weather_conditions,
        )
        previous_run_temperature_vector = current_run_temperature_vector
        logger.info(
            "Time: %s: Internal run complete with consistent conditions. Moving to the next time step.",
            date_and_time.strftime("%d/%m/%Y %H:%M:%S"),
        )

    return numpy.asarray(current_run_temperature_vector), system_data
