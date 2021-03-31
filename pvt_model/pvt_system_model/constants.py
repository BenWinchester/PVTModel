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
    "COLLECTOR_INPUT_TEMPERATURES",
    "CONVERGENT_SOLUTION_PRECISION",
    "DENSITY_OF_WATER",
    "EDGE_WIDTH",
    "EDGE_LENGTH",
    "FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR",
    "HEAT_CAPACITY_OF_WATER",
    "MAXIMUM_RECURSION_DEPTH",
    "NUMBER_OF_COLLECTORS",
    "NUSSELT_NUMBER",
    "STEFAN_BOLTZMAN_CONSTANT",
    "THERMAL_CONDUCTIVITY_OF_AIR",
    "THERMAL_CONDUCTIVITY_OF_WATER",
    "WARN_RECURSION_DEPTH",
    "ZERO_CELCIUS_OFFSET",
)

#############
# Constants #
#############

# @@@
# This constant is used in the equations in the model Gan sent through.
NUMBER_OF_COLLECTORS = 1
# NUMBER_OF_COLLECTORS = 19414.28
# NUMBER_OF_COLLECTORS = 200

# The temperature of absolute zero in Kelvin, used for converting Celcius to Kelvin and
# vice-a-versa.
ZERO_CELCIUS_OFFSET: float = 273.15

# Collector input temperatures.
COLLECTOR_INPUT_TEMPERATURES = {280, 300, 350, 400}
# The precision at which to calculate the convergent solution: 1 -> 0.1, 2 -> 0.01, etc.
CONVERGENT_SOLUTION_PRECISION = 1
# The default system temperature, used for instantiating runs.
DEFAULT_SYSTEM_TEMPERATURE = ZERO_CELCIUS_OFFSET + 30  # [K]
# The density of water, measured in kilograms per meter cubed.
DENSITY_OF_WATER: int = 1000
# The maximum size allowed for the width of edge segments, measured in meters.
EDGE_WIDTH = 0.005
# The maximum size allowed for the length of top and bottom-eg
EDGE_LENGTH = 0.005
# The free convective, heat-transfer coefficient of air. This varies, and potentially
# could be extended to the weather module and calculated on the fly depending on various
# environmental conditions etc.. This is measured in Watts per meter squared
# Kelvin.
FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR: int = 25
# The heat capacity of water, measured in Joules per kilogram Kelvin.
HEAT_CAPACITY_OF_WATER: int = 4182
# The initial temperature of the hot-water tank, at which it should be instantiated,
# measured in Kelvin.
INITIAL_TANK_TEMPERATURE = ZERO_CELCIUS_OFFSET + 34.75  # [K]
# The maximum recursion depth for which to run the model. Beyond this, an error will be
# raised.
MAXIMUM_RECURSION_DEPTH = 20
# The Nusselt number of the flow is given as 6 in Maria's paper.
NUSSELT_NUMBER: float = 4.36
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
# The recursion depth at which to raise a warning.
WARN_RECURSION_DEPTH = 10
