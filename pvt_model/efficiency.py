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

__all__ = (
    "dc_electrical",
    "dc_thermal",
    "dc_average",
    "dc_average_from_dc_values",
    "dc_weighted_average",
    "dc_weighted_average_from_dc_values",
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

    # * Compute the electrical demand covered.


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

    # * Compute the thermal demand covered.


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

    # * Determine the average output met by calling the other two functions.


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

    # * Determine the average output met by averagine the two input values.


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

    # * Determine the weighted average output met by calling the other two functions.
    # * From these, the demand covered values are computed, and a weighted average is
    # * determined.


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

    # * Determine the weighted average output met by averaging the two values in a
    # * weighted fashion.
