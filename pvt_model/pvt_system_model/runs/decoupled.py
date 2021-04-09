#!/usr/bin/python3.7
########################################################################################
# decoupled.py - Module for executing decoupled system runs.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The decoupled-run module for the PV-T model.

This module exposes methods needed to carry out a decoupled run of the system.

A decoupled run treates the PV-T collector in isolation, separated from the rest of the
system. Various data about the nature of the panel in this mode of operation is gathered
by calling out to the matrix module.

"""

import datetime
import logging

from typing import Dict, List, Tuple

import numpy

from .. import convergent_solver, weather
from ..pvt_panel import pvt

from ...__utils__ import OperatingMode, SystemData
from ..__utils__ import DivergentSolutionError
from ..constants import CONVERGENT_SOLUTION_PRECISION
from .__utils__ import system_data_from_run

__all__ = ("decoupled_run",)


def decoupled_run(
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
) -> Tuple[numpy.ndarray, Dict[float, SystemData]]:
    """
    Carries out a steady-state run of the system.

    """

    # Set up a holder for information about the system.
    system_data: Dict[float, SystemData] = dict()

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

    system_data[0] = system_data_from_run(
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
        current_run_temperature_vector = (
            convergent_solver.solve_temperature_vector_convergence_method(
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
        )
    except DivergentSolutionError as e:
        logger.error(
            "A divergent solution was when attempting to solve the system in steady-state:%s",
            str(e),
        )
        raise

    # Save the system data output and 2D profiles.
    system_data[1] = system_data_from_run(
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

    return numpy.asarray(current_run_temperature_vector), system_data
