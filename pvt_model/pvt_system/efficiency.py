#!/usr/bin/python3.7
########################################################################################
# efficiency.py - Calculates the efficiency profile and characteristics of the module.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The efficiency module for this PV-T model.

This module computes various efficiency characteristics for the PV-T systems, used to
quantify its efficacy, efficiency, and whether it can meet, and to what extent it meets,
the demands placed on it by end users.

NOTE: These functions can be called either at an individual time step, to compute the
instantaneous demand covered values, or with average values over some time interval,
such as over an entire year to compute the annual demand covered value.

"""

from .constants import HEAT_CAPACITY_OF_WATER
from .pvt_collector import pvt

__all__ = (
    "dc_electrical",
    "dc_thermal",
    "dc_average",
    "dc_average_from_dc_values",
    "dc_weighted_average",
    "dc_weighted_average_from_dc_values",
    "electrical_efficiency",
    "thermal_efficiency",
)


def dc_electrical(
    electrical_output: float, electrical_losses: float, electrical_demand: float
) -> float:
    """
    Determines the percentage of electrical demand covered.

    :param electrical_output:
        The electrical output of the system, measured in Watts.

    :param electrical_losses:
        Electrical losses within the system, such as water-pump use, measured in Watts.

    :param electrical_demand:
        The electrical demand of the system, measured in Watts.

    :return:
        The electrical demand covered as a value between 0 (corresponding to 0% demand
        covered) and 1 (corresponding to 100% demand covered).

    """

    return (electrical_output - electrical_losses) / electrical_demand


def dc_thermal(thermal_output: float, thermal_demand: float) -> float:
    """
    Determines the percentage of electrical demand covered.

    :param thermal_output:
        The thermal output of the system, measured in Watts.

    :param thermal_demand:
        The thermal demand of the system, measured in Watts.

    :return:
        The thermal demand covered as a value between 0 (corresponding to 0% demand
        covered) and 1 (corresponding to 100% demand covered).

    """

    return thermal_output / thermal_demand


def dc_average(
    electrical_output: float,
    electrical_losses: float,
    electrical_demand: float,
    thermal_output: float,
    thermal_demand: float,
) -> float:
    """
    Determines the average demand covered across both electrical and thermal outputs.

    :param electrical_output:
        The electrical output of the system, measured in Watts.

    :param electrical_losses:
        Electrical losses within the system, such as water-pump use, measured in Watts.

    :param electrical_demand:
        The electrical demand of the system, measured in Watts.

    :param thermal_output:
        The thermal output of the system, measured in Watts.

    :param thermal_demand:
        The thermal demand of the system, measured in Watts.

    :return:
        The un-weighted average of both the electrical and thermal demand-covered
        values, returned as a value between 0 (corresponding to 0% demand covered) and 1
        (corresponding to 100% demand covered).

    """

    return 0.5 * (
        dc_electrical(electrical_output, electrical_losses, electrical_demand)
        + dc_thermal(thermal_output, thermal_demand)
    )


def dc_average_from_dc_values(
    dc_electrical_value: float, dc_thermal_value: float
) -> float:
    """
    Determines the average demand covered across both electrical and thermal outputs.

    :param dc_electrical_value:
        The value for the electrical demand covered.

    :param dc_thermal_value:
        The value for the thermal demand covered.

    :return:
        The un-weighted average of both the electrical and thermal demand-covered
        values, returned as a value between 0 (corresponding to 0% demand covered) and 1
        (corresponding to 100% demand covered).

    """

    return 0.5 * (dc_electrical_value + dc_thermal_value)


def dc_weighted_average(
    electrical_output: float,
    electrical_losses: float,
    electrical_demand: float,
    thermal_output: float,
    thermal_demand: float,
) -> float:
    """
    Determines weighted-average demand covered over both electrical and thermal outputs.

    :param electrical_output:
        The electrical output of the system, measured in Watts.

    :param electrical_losses:
        Electrical losses within the system, such as water-pump use, measured in Watts.

    :param electrical_demand:
        The electrical demand of the system, measured in Watts.

    :param thermal_output:
        The thermal output of the system, measured in Watts.

    :param thermal_demand:
        The thermal demand of the system, measured in Watts.

    :return:
        The weighted average of both the electrical and thermal demand-covered values,
        returned as a value between 0 (corresponding to 0% demand covered) and 1
        (corresponding to 100% demand covered).

    """

    return (
        electrical_demand
        * dc_electrical(electrical_output, electrical_losses, electrical_demand)
        + thermal_demand * dc_thermal(thermal_output, thermal_demand)
    ) / (electrical_demand + thermal_demand)


def dc_weighted_average_from_dc_values(
    dc_electrical_value: float,
    dc_thermal_value: float,
    electrical_demand: float,
    thermal_demand: float,
) -> float:
    """
    Determines weighted-average demand covered over both electrical and thermal outputs.

    :param dc_electrical_value:
        The value for the electrical demand covered.

    :param dc_thermal_value:
        The value for the thermal demand covered.

    :param electrical_demand:
        The electrical demand of the system, measured in Watts.

    :param thermal_demand:
        The thermal demand of the system, measured in Watts.

    :return:
        The weighted average of both the electrical and thermal demand-covered values,
        returned as a value between 0 (corresponding to 0% demand covered) and 1
        (corresponding to 100% demand covered).

    """

    return (
        electrical_demand * dc_electrical_value + thermal_demand * dc_thermal_value
    ) / (electrical_demand + thermal_demand)


def electrical_efficiency(
    pvt_collector: pvt.PVT,
    temperature: float,
) -> float:
    """
    Computes the electrical efficiency of a PV cell or element.

    :param pvt_collector:
        The PVT collector instance being modelled.

    :param temperature:
        The temperature of the PV layer, segment, or element, measured in Kelvin.

    :return:
        The electrical efficiency.

    """

    return pvt_collector.pv.reference_efficiency * (
        1
        - pvt_collector.pv.thermal_coefficient
        * (temperature - pvt_collector.pv.reference_temperature)
    )


def thermal_efficiency(
    area: float, mass_flow_rate: float, solar_irradiance: float, temperature_gain: float
) -> float:
    """
    Compute the thermal efficiency.

    :param area:
        The area of the panel, measured in meters squared.

    :param mass_flow_rate:
        The mass flow rate of HTF through the absorber.

    :param solar_irradiance:
        The solar irradiance, measured in Watts per meter squared.

    :param temperature_gain:
        The temperature gain across the panel, measured in Kelvin.

    :return:
        The thermal efficiency, based on the input parameters, for a PV-T absorber.

    """

    # Compute the thermal efficiency of the absorber.
    thermal_input: float = area * solar_irradiance
    thermal_output: float = mass_flow_rate * HEAT_CAPACITY_OF_WATER * temperature_gain

    return thermal_output / thermal_input
