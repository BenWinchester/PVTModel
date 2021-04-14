#!/usr/bin/python3.7
########################################################################################
# physics_utils.py - Utility module containing Physics equations.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The physics utility module for the PVT model's PVT panel component.

"""

from typing import List, Union

from numpy import ndarray

from .. import physics_utils
from . import pvt

from ..__utils__ import WeatherConditions

__all__ = ("insulation_thermal_resistance",)


def insulation_thermal_resistance(
    best_guess_temperature_vector: Union[List[float], ndarray],
    pvt_panel: pvt.PVT,
    source_index: int,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Returns the thermal resistance between the back layer of the absorber and air.

    Insulation on the back of the PV-T absorber causes there to be some thermal
    resistance to the heat transfer out of the back of the thermal absorber. This
    value is computed here and returned.

    :param best_guess_temperature_vector:
        The best-guess at the temperature vector for the system at the current time
        step.

    :param pvt_panel:
        The :class:`pvt.PVT` instance representing the pvt panel being modelled.

    :param source_index:
        The index of the segment emitting to the environment.

    :param weather_conditions:
        The weather conditions at the current time step.

    :return:
        The thermal resistance between the back layer of the absorber and the
        surrounding air, measured in meter squared Kelvin per Watt.

    """

    return (
        pvt_panel.insulation.thickness / pvt_panel.insulation.conductivity
        + 1
        / physics_utils.free_heat_transfer_coefficient_of_air(
            pvt_panel, best_guess_temperature_vector[source_index], weather_conditions
        )
    )
