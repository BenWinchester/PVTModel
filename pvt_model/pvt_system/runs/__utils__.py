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
from ..pvt_collector import pvt

from ...__utils__ import OperatingMode, SystemData, TemperatureName
from ..__utils__ import (
    ProgrammerJudgementFault,
    WeatherConditions,
)
from ..constants import ZERO_CELCIUS_OFFSET
from ..physics_utils import reduced_temperature


__all__ = ("system_data_from_run",)


def _average_layer_temperature(
    pvt_collector: pvt.PVT,
    temperature_name: TemperatureName,
    temperature_vector: Union[List[float], numpy.ndarray],
) -> float:
    """
    Determines the average temperature for a layer.

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
        if index_handler.temperature_name_from_index(value_index, pvt_collector)
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
    number_of_x_elements: int,
    pvt_collector: pvt.PVT,
    temperature_name: TemperatureName,
    temperature_vector: Union[List[float], numpy.ndarray],
) -> Dict[str, float]:
    """
    Returns a map between coordinates and temperature in celcius for a layer.

    :param number_of_pipes:
        The number of pipes in the absorber.

    :param number_of_x_elements:
        The number of elements in a single row of the absorber.

    :param pvt_collector:
        The PVT panel being modelled.

    :param temperature_name:
        The name of the temperature for which to determine the average.

    :param temperature_vector:
        The temperature vector from which the average temperature of the given layer
        should be extracted.

    :return:
        A mapping between coordinates, expressed as a string, and the temperature of
        each element, for the layer specified. The values of the temperatures are
        expressed in Celcius.

    """

    if temperature_name in [
        TemperatureName.upper_glass,
        TemperatureName.glass,
        TemperatureName.pv,
        TemperatureName.absorber,
    ]:
        layer_temperature_map: Dict[str, float] = {
            str(element_coordinates): temperature_vector[
                index_handler.index_from_element_coordinates(
                    number_of_x_elements,
                    pvt_collector,
                    temperature_name,
                    element.x_index,
                    element.y_index,
                )
            ]
            - ZERO_CELCIUS_OFFSET
            for element_coordinates, element in pvt_collector.elements.items()
        }
    elif temperature_name in [
        TemperatureName.pipe,
        TemperatureName.htf,
        TemperatureName.htf_in,
        TemperatureName.htf_out,
    ]:
        layer_temperature_map = {
            str(element_coordinates): temperature_vector[
                index_handler.index_from_pipe_coordinates(
                    number_of_pipes,
                    number_of_x_elements,
                    element.pipe_index,  # type: ignore
                    pvt_collector,
                    temperature_name,
                    element.y_index,
                )
            ]
            - ZERO_CELCIUS_OFFSET
            for element_coordinates, element in pvt_collector.elements.items()
            if element.pipe
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
    number_of_x_elements: int,
    operating_mode: OperatingMode,
    pvt_collector: pvt.PVT,
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

    :param number_of_x_elements:
        The number of x elements included in the absorber.

    :param operating_mode:
        The operating mode for the run.

    :param pvt_collector:
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
    if any({element.upper_glass for element in pvt_collector.elements.values()}):
        average_upper_glass_temperature: Optional[float] = _average_layer_temperature(
            pvt_collector,
            TemperatureName.upper_glass,
            temperature_vector,
        )
        temperature_map_upper_glass_layer: Optional[
            Dict[str, float]
        ] = _layer_temperature_profile(
            number_of_pipes,
            number_of_x_elements,
            pvt_collector,
            TemperatureName.upper_glass,
            temperature_vector,
        )
    else:
        average_upper_glass_temperature = None
        temperature_map_upper_glass_layer = None

    if any({element.glass for element in pvt_collector.elements.values()}):
        average_glass_temperature: Optional[float] = _average_layer_temperature(
            pvt_collector,
            TemperatureName.glass,
            temperature_vector,
        )
        temperature_map_glass_layer: Optional[
            Dict[str, float]
        ] = _layer_temperature_profile(
            number_of_pipes,
            number_of_x_elements,
            pvt_collector,
            TemperatureName.glass,
            temperature_vector,
        )
    else:
        average_glass_temperature = None
        temperature_map_glass_layer = None

    average_pv_temperature = _average_layer_temperature(
        pvt_collector,
        TemperatureName.pv,
        temperature_vector,
    )
    temperature_map_pv_layer = _layer_temperature_profile(
        number_of_pipes,
        number_of_x_elements,
        pvt_collector,
        TemperatureName.pv,
        temperature_vector,
    )

    average_absorber_temperature = _average_layer_temperature(
        pvt_collector,
        TemperatureName.absorber,
        temperature_vector,
    )
    temperature_map_absorber_layer = _layer_temperature_profile(
        number_of_pipes,
        number_of_x_elements,
        pvt_collector,
        TemperatureName.absorber,
        temperature_vector,
    )

    average_pipe_temperature = _average_layer_temperature(
        pvt_collector,
        TemperatureName.pipe,
        temperature_vector,
    )
    temperature_map_pipe_layer = _layer_temperature_profile(
        number_of_pipes,
        number_of_x_elements,
        pvt_collector,
        TemperatureName.pipe,
        temperature_vector,
    )

    average_bulk_water_temperature = _average_layer_temperature(
        pvt_collector,
        TemperatureName.htf,
        temperature_vector,
    )
    temperature_map_bulk_water_layer = _layer_temperature_profile(
        number_of_pipes,
        number_of_x_elements,
        pvt_collector,
        TemperatureName.htf,
        temperature_vector,
    )

    # Set various parameters depending on the operating mode of the model.
    # Set the variables that depend on a coupled vs decoupled system.
    if operating_mode.coupled:
        exchanger_temperature_drop: Optional[float] = (
            temperature_vector[
                index_handler.index_from_temperature_name(
                    pvt_collector,
                    TemperatureName.tank_out,
                )
            ]
            - temperature_vector[
                index_handler.index_from_temperature_name(
                    pvt_collector,
                    TemperatureName.tank_in,
                )
            ]
            if temperature_vector[
                index_handler.index_from_temperature_name(
                    pvt_collector,
                    TemperatureName.tank_in,
                )
            ]
            > temperature_vector[
                index_handler.index_from_temperature_name(
                    pvt_collector,
                    TemperatureName.tank,
                )
            ]
            else 0
        )
        tank_temperature: Optional[float] = (
            temperature_vector[
                index_handler.index_from_temperature_name(
                    pvt_collector,
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
            pvt_collector,
            TemperatureName.collector_in,
        )
    ]
    collector_output_temperature = temperature_vector[
        index_handler.index_from_temperature_name(
            pvt_collector,
            TemperatureName.collector_out,
        )
    ]

    # Determine the reduced temperature of the system.
    if weather_conditions.irradiance > 0:
        electrical_efficiency: Optional[float] = efficiency.electrical_efficiency(
            pvt_collector, average_pv_temperature
        )
        reduced_system_temperature: Optional[float] = reduced_temperature(
            weather_conditions.ambient_temperature,
            average_bulk_water_temperature,
            weather_conditions.irradiance,
        )
        thermal_efficiency: Optional[float] = efficiency.thermal_efficiency(
            pvt_collector.area,
            pvt_collector.absorber.mass_flow_rate,
            weather_conditions.irradiance,
            collector_output_temperature - collector_input_temperature,
        )
    else:
        electrical_efficiency = None
        reduced_system_temperature = None
        thermal_efficiency = None

    # Return the system data.
    return SystemData(
        date=date.strftime("%d/%m/%Y"),
        time=formatted_time,
        upper_glass_temperature=average_upper_glass_temperature - ZERO_CELCIUS_OFFSET
        if average_upper_glass_temperature is not None
        else None,
        glass_temperature=average_glass_temperature - ZERO_CELCIUS_OFFSET
        if average_glass_temperature is not None
        else None,
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
        layer_temperature_map_upper_glass=temperature_map_upper_glass_layer
        if save_2d_output
        else None,
        electrical_efficiency=electrical_efficiency,
        reduced_collector_temperature=reduced_system_temperature,
        thermal_efficiency=thermal_efficiency,
        solar_irradiance=weather_conditions.irradiance,
    )
