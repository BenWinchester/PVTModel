#!/usr/bin/python3.7
########################################################################################
# constants.py - Module for holding and exporting fundamental constants.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Physics utility module for the PV-T panel component.

This module contains formulae for calculating Physical values in relation to the PVT
panel.

"""

__all__ = (
    "DENSITY_OF_WATER",
    "FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR",
    "HEAT_CAPACITY_OF_WATER",
    "INITIAL_SYSTEM_TEMPERATURE_VECTOR",
    "NUSSELT_NUMBER",
    "STEFAN_BOLTZMAN_CONSTANT",
    "THERMAL_CONDUCTIVITY_OF_AIR",
    "THERMAL_CONDUCTIVITY_OF_WATER",
    "ZERO_CELCIUS_OFFSET",
)

#############
# Constants #
#############

# The temperature of absolute zero in Kelvin, used for converting Celcius to Kelvin and
# vice-a-versa.
ZERO_CELCIUS_OFFSET: float = 273.15

# The density of water, measured in kilograms per meter cubed.
DENSITY_OF_WATER: int = 1000
# The free convective, heat-transfer coefficient of air. This varies, and potentially
# could be extended to the weather module and calculated on the fly depending on various
# environmental conditions etc.. This is measured in Watts per meter squared
# Kelvin.
FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR: int = 25
# The heat capacity of water, measured in Joules per kilogram Kelvin.
HEAT_CAPACITY_OF_WATER: int = 4182
# The initial temperature for the system to be instantiated at, measured in Kelvin.
# The `tuple` contains the glass, pv, collector, bulk-water, and tank initial
# temperatures.
INITIAL_SYSTEM_TEMPERATURE_VECTOR = (
    ZERO_CELCIUS_OFFSET + 20,  # Var: T_Glass / K
    ZERO_CELCIUS_OFFSET + 35,  # Var: T_PV / K
    ZERO_CELCIUS_OFFSET + 34,  # Var: T_Collector / K
    ZERO_CELCIUS_OFFSET + 34,  # Var: T_BulkWater / K
    ZERO_CELCIUS_OFFSET + 30,  # Var: T_Tank / K
)  # [K]
# The initial temperature of the hot-water tank, at which it should be instantiated,
# measured in Kelvin.
INITIAL_TANK_TEMPERATURE = ZERO_CELCIUS_OFFSET + 34.75  # [K]
# The Nusselt number of the flow is given as 6 in Maria's paper.
NUSSELT_NUMBER: float = 6
# The Stefan-Boltzman constant, given in Watts per meter squared Kelvin to the four.
STEFAN_BOLTZMAN_CONSTANT: float = 5.670374419 * (10 ** (-8))
# The convective, heat-transfer coefficienct of water. This varies (a LOT), and is
# measured in units of Watts per meter squared Kelvin.
# This is determined by the following formula:
#   Nu = h_w * D / k_w
# where D is the diameter of the pipe, in meters, and k_w is the thermal conductivity of
# water.
# @@@ I think a constant should be inserted here.

# The thermal conductivity of air is measured in Watts per meter Kelvin.
# ! This is defined at 273 Kelvin.
THERMAL_CONDUCTIVITY_OF_AIR: float = 0.024
# The thermal conductivity of water is obtained from
# http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/thrcn.html
THERMAL_CONDUCTIVITY_OF_WATER: float = 0.5918  # [W/m*K]
