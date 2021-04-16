#!/usr/bin/python3.7
########################################################################################
# constants.py - Module for holding and exporting fundamental constants.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Constants utility module for the pvt model component.

This module contains various physical constants and internal model-specific constants
which are accessed across multiple files.

"""

import datetime
import pytz

__all__ = (
    "ACCELERATION_DUE_TO_GRAVITY",
    "CONVERGENT_SOLUTION_PRECISION",
    "DEFAULT_INITIAL_DATE_AND_TIME",
    "DENSITY_OF_WATER",
    "EDGE_WIDTH",
    "EDGE_LENGTH",
    "HEAT_CAPACITY_OF_WATER",
    "HOT_WATER_DEMAND_TEMP",
    "INTERNAL_HOUSEHOLD_AMBIENT_TEMPERATURE",
    "MAXIMUM_RECURSION_DEPTH",
    "NUMBER_OF_COLLECTORS",
    "SPECIFIC_GAS_CONSTANT_OF_AIR",
    "STEFAN_BOLTZMAN_CONSTANT",
    "THERMAL_CONDUCTIVITY_OF_WATER",
    "WARN_RECURSION_DEPTH",
    "ZERO_CELCIUS_OFFSET",
)

#############
# Constants #
#############

# This scales the entire system's size - NOTE: Now depreciated.
NUMBER_OF_COLLECTORS = 1
# NUMBER_OF_COLLECTORS = 19414.28
# NUMBER_OF_COLLECTORS = 200

# The temperature of absolute zero in Kelvin, used for converting Celcius to Kelvin and
# vice-a-versa.
ZERO_CELCIUS_OFFSET: float = 273.15

# The acceleration due to gravity measured in meters per second squared.
ACCELERATION_DUE_TO_GRAVITY: float = 9.81

# The precision at which to calculate the convergent solution: 1 -> 0.1, 2 -> 0.01, etc.
CONVERGENT_SOLUTION_PRECISION = 1

# The initial date and time for the simultion to run from.
DEFAULT_INITIAL_DATE_AND_TIME = datetime.datetime(2005, 1, 1, 0, 0, tzinfo=pytz.UTC)

# The default system temperature, used for instantiating runs.
DEFAULT_SYSTEM_TEMPERATURE = ZERO_CELCIUS_OFFSET + 18.41  # [K]

# The density of water, measured in kilograms per meter cubed.
DENSITY_OF_WATER: int = 1000

# The maximum size allowed for the width of edge elements, measured in meters.
EDGE_WIDTH = 0.005

# The maximum size allowed for the length of top and bottom-eg
EDGE_LENGTH = 0.005

# The heat capacity of air, measured in Joules per kilogram Kelvin.
HEAT_CAPACITY_OF_AIR: int = 700

# The heat capacity of water, measured in Joules per kilogram Kelvin.
HEAT_CAPACITY_OF_WATER: int = 4182

# The initial temperature of the hot-water tank, at which it should be instantiated,
# measured in Kelvin.
# The temperature of hot-water required by the end-user, measured in Kelvin.
HOT_WATER_DEMAND_TEMP = 60 + ZERO_CELCIUS_OFFSET

# The average temperature of the air surrounding the tank, which is internal to the
# household, measured in Kelvin.
INTERNAL_HOUSEHOLD_AMBIENT_TEMPERATURE = ZERO_CELCIUS_OFFSET + 20  # [K]

# The maximum recursion depth for which to run the model. Beyond this, an error will be
# raised.
MAXIMUM_RECURSION_DEPTH = 20

# The gas constant of air in SI units of Joules per kilogram Kelvin.
SPECIFIC_GAS_CONSTANT_OF_AIR = 287.05  # [J/kg*K]

# The Stefan-Boltzman constant, given in Watts per meter squared Kelvin to the four.
STEFAN_BOLTZMAN_CONSTANT: float = 5.670374419 * (10 ** (-8))

# The thermal conductivity of water is obtained from
# http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/thrcn.html
THERMAL_CONDUCTIVITY_OF_WATER: float = 0.5918  # [W/m*K]

# The recursion depth at which to raise a warning.
WARN_RECURSION_DEPTH = 10
