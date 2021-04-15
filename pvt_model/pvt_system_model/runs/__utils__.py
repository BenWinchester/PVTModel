#!/usr/bin/python3.7
########################################################################################
# __utils__.py - The utility module for the runs component of the pvt system model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The utility module for the PV-T system model's runs component.

This module contains common functionality, strucutres, and types, to be used by the
various modules throughout the PVT model.

"""

import datetime

from typing import Dict, List, Optional, Set, Union

import numpy

from .. import efficiency, index_handler
from ..pvt_panel import pvt

from ...__utils__ import OperatingMode, SystemData, TemperatureName
from ..__utils__ import (
    ProgrammerJudgementFault,
    WeatherConditions,
)
from ..constants import ZERO_CELCIUS_OFFSET
from ..physics_utils import reduced_temperature
from ..pvt_panel.segment import Segment, SegmentCoordinates


__all__ = ("system_data_from_run",)


def _average_layer_temperature(
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    temperature_name: TemperatureName,
    temperature_vector: Union[List[float], numpy.ndarray],
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


def _layer_temperature_profile(
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    segments: Dict[SegmentCoordinates, Segment],
    temperature_name: TemperatureName,
    temperature_vector: Union[List[float], numpy.ndarray],
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
                    segment.pipe_index,  # type: ignore
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


def system_data_from_run(
    date: datetime.date,
    initial_date_and_time: datetime.datetime,
    number_of_pipes: int,
    number_of_x_segments: int,
    number_of_y_segments: int,
    operating_mode: OperatingMode,
    pvt_panel: pvt.PVT,
    save_2d_output: bool,
    temperature_vector: Union[List[float], numpy.ndarray],
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
        formatted_time: Optional[str] = (
            str((date.day - initial_date_and_time.day) * 24 + time.hour)
            + time.strftime("%H:%M:%S")[2:]
        )
    else:
        formatted_time = None

    # Compute variables in common to both.
    collector_input_temperature = temperature_vector[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector_in,
        )
    ]
    collector_output_temperature = temperature_vector[
        index_handler.index_from_temperature_name(
            number_of_pipes,
            number_of_x_segments,
            number_of_y_segments,
            TemperatureName.collector_out,
        )
    ]

    # Determine the reduced temperature of the system.
    if weather_conditions.irradiance > 0:
        reduced_system_temperature: Optional[float] = reduced_temperature(
            weather_conditions.ambient_temperature,
            average_bulk_water_temperature,
            weather_conditions.irradiance,
        )
        thermal_efficiency: Optional[float] = efficiency.thermal_efficiency(
            pvt_panel.area,
            pvt_panel.absorber.mass_flow_rate,
            weather_conditions.irradiance,
            collector_output_temperature - collector_input_temperature,
        )
    else:
        reduced_system_temperature = None
        thermal_efficiency = None

    # Return the system data.
    return SystemData(
        date=date.strftime("%d/%m/%Y"),
        time=formatted_time,
        glass_temperature=average_glass_temperature - ZERO_CELCIUS_OFFSET,
        pv_temperature=average_pv_temperature - ZERO_CELCIUS_OFFSET,
        absorber_temperature=average_absorber_temperature - ZERO_CELCIUS_OFFSET,
        collector_input_temperature=collector_input_temperature - ZERO_CELCIUS_OFFSET,
        collector_output_temperature=collector_output_temperature - ZERO_CELCIUS_OFFSET,
        pipe_temperature=average_pipe_temperature - ZERO_CELCIUS_OFFSET,
        bulk_water_temperature=average_bulk_water_temperature - ZERO_CELCIUS_OFFSET,
        ambient_temperature=weather_conditions.ambient_temperature
        - ZERO_CELCIUS_OFFSET,
        exchanger_temperature_drop=exchanger_temperature_drop,
        sky_temperature=weather_conditions.sky_temperature - ZERO_CELCIUS_OFFSET,
        tank_temperature=tank_temperature,
        collector_temperature_gain=collector_output_temperature
        - collector_input_temperature,
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
        reduced_collector_temperature=reduced_system_temperature,
        thermal_efficiency=thermal_efficiency,
        solar_irradiance=weather_conditions.irradiance,
    )
