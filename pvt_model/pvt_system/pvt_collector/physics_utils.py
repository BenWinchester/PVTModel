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

import math

from typing import List, Union

from numpy import ndarray

from .. import physics_utils
from . import pvt

from ...__utils__ import BColours, ProgrammerJudgementFault
from ..__utils__ import WeatherConditions

__all__ = (
    "glass_glass_air_gap_resistance",
    "glass_pv_air_gap_resistance",
    "insulation_thermal_resistance",
)


def _conductive_heat_transfer_coefficient_with_gap(
    air_gap_thickness: float,
    average_surface_temperature: float,
    pvt_collector: pvt.PVT,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Computes the conductive heat transfer between the two layers, measured in W/m^2*K.

    The value computed is positive if the heat transfer is from the source to the
    destination, as determined by the arguments, and negative if the flow of heat is
    the reverse of what is implied via the parameters.

    The value for the heat transfer is returned in Watts.

    :param air_gap_thickness:
        The thickness of the air gap between the PV and glass layers.

    :param average_surface_temperature:
        The average temperature of the surfaces across which the heat transfer is taking
        place.

    :param pvt_collector:
        The PVT panel being modelled.

    :param weather_conditions:
        The weather conditions at the time step being investigated.

    :return:
        The heat transfer coefficient, in Watts per meter squared Kelvin, between the
        two layers.

    """

    if pvt_collector.air_gap_thickness is None:
        raise ProgrammerJudgementFault(
            "{}Conductive heat transfer requested with no air gap.{}".format(
                BColours.FAIL, BColours.ENDC
            )
        )
    air_gap_rayleigh_number = physics_utils.rayleigh_number(
        pvt_collector.air_gap_thickness,
        average_surface_temperature,
        pvt_collector.tilt_in_radians,
        weather_conditions,
    )

    first_corrective_term: float = 1 - 1708 / air_gap_rayleigh_number
    first_corrective_term = max(first_corrective_term, 0)

    second_corrective_term: float = 1 - (
        1708 * (math.sin(1.8 * pvt_collector.tilt_in_radians)) ** 1.6
    ) / (air_gap_rayleigh_number * math.cos(pvt_collector.tilt_in_radians))
    second_corrective_term = max(second_corrective_term, 0)

    third_corrective_term: float = (
        (air_gap_rayleigh_number * math.cos(pvt_collector.tilt_in_radians)) / 5830
    ) ** 0.33 - 1
    third_corrective_term = max(third_corrective_term, 0)

    return (
        weather_conditions.thermal_conductivity_of_air  # [W/m*K]
        / air_gap_thickness  # [m]
    ) * (
        1
        + 1.44 * first_corrective_term * second_corrective_term
        + third_corrective_term
    )


def glass_absorber_air_gap_resistance(
    pvt_collector: pvt.PVT,
    surface_temperature: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Returns the thermal resistance of the air gap between the absorber and glass layers.

    :param pvt_collector:
        The :class:`pvt.PVT` instance representing the pvt panel being modelled.

    :param surface_temperature:
        The average temperature of the two surfaces either side of the air gap.

    :param weather_conditions:
        The weather conditions at the time step being investigated.

    :return:
        The thermal resistance, measured in Kelvin meter squared per Watt.

    """

    if pvt_collector.glass is None or pvt_collector.air_gap_thickness is None:
        raise ProgrammerJudgementFault(
            "{}Resistance across an air gap could not be computed due to no ".format(
                BColours.FAIL
            )
            + "glass layer being present.{}".format(BColours.ENDC)
        )

    return (
        pvt_collector.eva.thickness / pvt_collector.eva.conductivity
        + pvt_collector.glass.thickness / pvt_collector.glass.conductivity
        + pvt_collector.absorber.thickness / (2 * pvt_collector.absorber.conductivity)
        + pvt_collector.glass.thickness / (2 * pvt_collector.glass.conductivity)
        + 1
        / _conductive_heat_transfer_coefficient_with_gap(
            pvt_collector.air_gap_thickness,
            surface_temperature,
            pvt_collector,
            weather_conditions,
        )
    )


def glass_glass_air_gap_resistance(
    pvt_collector: pvt.PVT,
    surface_temperature: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Returns the thermal resistance of the air gap between two glass layers.

    :param pvt_collector:
        The :class:`pvt.PVT` instance representing the pvt panel being modelled.

    :param surface_temperature:
        The average temperature of the two surfaces either side of the air gap.

    :param weather_conditions:
        The weather conditions at the time step being investigated.

    :return:
        The thermal resistance, measured in Kelvin meter squared per Watt.

    """

    if (
        pvt_collector.glass is None
        or pvt_collector.air_gap_thickness is None
        or pvt_collector.upper_glass is None
    ):
        raise ProgrammerJudgementFault(
            "{}Resistance across an air gap could not be computed due to no ".format(
                BColours.FAIL
            )
            + "glass layer being present.{}".format(BColours.ENDC)
        )

    return (
        pvt_collector.eva.thickness / pvt_collector.eva.conductivity
        + pvt_collector.upper_glass.thickness / pvt_collector.upper_glass.conductivity
        + pvt_collector.glass.thickness / (2 * pvt_collector.glass.conductivity)
        + pvt_collector.upper_glass.thickness
        / (2 * pvt_collector.upper_glass.conductivity)
        + 1
        / _conductive_heat_transfer_coefficient_with_gap(
            pvt_collector.air_gap_thickness,
            surface_temperature,
            pvt_collector,
            weather_conditions,
        )
    )


def glass_pv_air_gap_resistance(
    pvt_collector: pvt.PVT,
    surface_temperature: float,
    weather_conditions: WeatherConditions,
) -> float:
    """
    Returns the thermal resistance of the air gap between the PV and glass layers.

    :param pvt_collector:
        The :class:`pvt.PVT` instance representing the pvt panel being modelled.

    :param surface_temperature:
        The average temperature of the two surfaces either side of the air gap.

    :param weather_conditions:
        The weather conditions at the time step being investigated.

    :return:
        The thermal resistance, measured in Kelvin meter squared per Watt.

    """

    if pvt_collector.glass is None or pvt_collector.air_gap_thickness is None:
        raise ProgrammerJudgementFault(
            "{}Resistance across an air gap could not be computed due to no ".format(
                BColours.FAIL
            )
            + "glass layer being present.{}".format(BColours.ENDC)
        )

    return (
        pvt_collector.eva.thickness / pvt_collector.eva.conductivity
        + pvt_collector.glass.thickness / pvt_collector.glass.conductivity
        + pvt_collector.pv.thickness / (2 * pvt_collector.pv.conductivity)
        + pvt_collector.glass.thickness / (2 * pvt_collector.glass.conductivity)
        + 1
        / _conductive_heat_transfer_coefficient_with_gap(
            pvt_collector.air_gap_thickness,
            surface_temperature,
            pvt_collector,
            weather_conditions,
        )
    )


def insulation_thermal_resistance(
    best_guess_temperature_vector: Union[List[float], ndarray],
    pvt_collector: pvt.PVT,
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

    :param pvt_collector:
        The :class:`pvt.PVT` instance representing the pvt panel being modelled.

    :param source_index:
        The index of the element emitting to the environment.

    :param weather_conditions:
        The weather conditions at the current time step.

    :return:
        The thermal resistance between the back layer of the absorber and the
        surrounding air, measured in meter squared Kelvin per Watt.

    """

    return (
        pvt_collector.insulation.thickness / pvt_collector.insulation.conductivity
        + 1
        / physics_utils.free_heat_transfer_coefficient_of_air(
            pvt_collector,
            best_guess_temperature_vector[source_index],
            weather_conditions,
        )
    )
