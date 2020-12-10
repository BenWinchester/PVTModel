#!/usr/bin/python3.7
########################################################################################
# __utils__.py - The utility module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The utility module for the PV-T model.

This module contains common functionality, strucutres, and types, to be used by the
various modules throughout the PVT model.

"""

import enum
import logging
import os

from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

__all__ = (
    # Exceptions
    "InternalError",
    "InvalidDataError",
    "MissingDataError",
    "MissingParametersError",
    "ResolutionMismatchError",
    # Dataclasses, Enums and Named Tuples
    "BackLayerParameters",
    "CollectorParameters",
    "FileType",
    "GraphDetail",
    "LayerParameters",
    "OpticalLayerParameters",
    "ProgrammerJudgementFault",
    "PVParameters",
    "WeatherConditions",
    # Helper functions
    "get_logger",
    "read_yaml",
    # Constants
    "FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR",
    "HEAT_CAPACITY_OF_WATER",
    "LOGGER_NAME",
    "NUSSELT_NUMBER",
    "STEFAN_BOLTZMAN_CONSTANT",
    "THERMAL_CONDUCTIVITY_OF_AIR",
    "THERMAL_CONDUCTIVITY_OF_WATER",
    "WIND_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT",
    "ZERO_CELCIUS_OFFSET",
)


#############
# Constants #
#############


LOGGER_NAME = "my_first_pvt_model"

# The Stefan-Boltzman constant, given in Watts per meter squared Kelvin to the four.
STEFAN_BOLTZMAN_CONSTANT: float = 5.670374419 * (10 ** (-8))

# The heat capacity of water, measured in Joules per kilogram Kelvin.
HEAT_CAPACITY_OF_WATER: int = 4182

# The free convective, heat-transfer coefficient of air. This varies, and potentially
# could be extended to the weather module and calculated on the fly depending on various
# environmental conditions etc.. This is measured in Watts per meter squared
# Kelvin.
FREE_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT_OF_AIR: int = 25

# The convective, heat-transfer coefficienct of water. This varies (a LOT), and is
# measured in units of Watts per meter squared Kelvin.
# This is determined by the following formula:
#   Nu = h_w * D / k_w
# where D is the diameter of the pipe, in meters, and k_w is the thermal conductivity of
# water.
# The thermal conductivity of water is obtained from
# http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/thrcn.html
THERMAL_CONDUCTIVITY_OF_WATER: float = 0.6  # [W/m*K]

# The wind convective heat transfer coefficient. This should be temperature dependant,
# @@@ Improve this.
# This should be measured in Watts per Kelvin.
WIND_CONVECTIVE_HEAT_TRANSFER_COEFFICIENT = 5

# The thermal conductivity of air is measured in Watts per meter Kelvin.
# ! This is defined at 273 Kelvin.
THERMAL_CONDUCTIVITY_OF_AIR: float = 0.024

# The temperature of absolute zero in Kelvin, used for converting Celcius to Kelvin and
# vice-a-versa.
ZERO_CELCIUS_OFFSET: float = 273.15

# The Nusselt number of the flow is given as 6 in Maria's paper.
NUSSELT_NUMBER: float = 6


##############
# Exceptions #
##############


class InternalError(Exception):
    """
    Used for internal error handling and catching where no information is needed.

    """


class InvalidDataError(Exception):
    """
    Raised when some proviced data is of the wrong format.

    """

    def __init__(self, data_file_name: str, message: str) -> None:
        """
        Instantiate an Invalid Data Error.

        :param data_file_name:
            The name of the data file containing the invalid data entry.

        :param message:
            An error message to be appended.

        """

        super().__init__(
            "Invalid data error. "
            "The file '{}' contained data of the wrong format: {}".format(
                data_file_name, message
            )
        )


class MissingDataError(Exception):
    """
    Raised when data is requested from a class that is missing.

    """

    def __init__(self, message: str) -> None:
        """
        Instantiate a Missing Data Error.

        :param message:
            The message to append for the user.

        """

        super().__init__(
            "Data requested from a class could not be found: {}".format(message)
        )


class MissingParametersError(Exception):
    """
    Raised when not all parameters have been specified that are needed to instantiate.

    """

    def __init__(self, class_name: str, message: str) -> None:
        """
        Instantiate a missing parameters error.

        :param class_name:
            The class for which insufficient parameters were specified.

        :param message:
            An appended message to display to the user.

        """

        super().__init__(
            f"Missing parameters when initialising a '{class_name}' class: {message}."
        )


class ResolutionMismatchError(Exception):
    """
    Raised when an error occurs attempting to match up data and simulation resolutions.

    """

    def __init__(self, message: str) -> None:
        """
        Instantiate a resolution mismatch error.

        :param message:
            An appended message to display to the user.

        """

        super().__init__(
            "Due to the nature of the load data, an integer multiple of resolutions, "
            "or divsions of resolutions, must be supplied with the '--resolution' or "
            "'-r' flag.\nI appreciate that this is poor coding, but at least I took "
            "the time to write a custom exception for it :p .\n Error message: "
            f"{message}"
        )


class ProgrammerJudgementFault(Exception):
    """
    Raised when an error is hit due to poor programming.

    """

    def __init__(self, message: str) -> None:
        """
        Instantiate a programmer judgement fault error.

        :param message:
            A message to append when displaying the error to the user.

        """

        super().__init__(f"A programmer judgement fault has occurred: {message}")


##############################
# Functions and Data Classes #
##############################


class GraphDetail(enum.Enum):
    """
    The level of detail to go into when graphing.

    .. attribute:: highest
        The highest level of detail - all data points are plotted.

    .. attribute:: high
        A "high" level of detail, to be determined by the analysis script.

    .. attribute:; medium
        A "medium" level of detail, to be determined by the analysis script.

    .. attribute:: low
        A "low" level of detail, to be determined by the analysis script.

    .. attribute:: lowest
        The lowest level of detail, with points only every half an hour.

    """

    highest = 0
    high = 1
    medium = 2
    low = 3
    lowest = 4


class FileType(enum.Enum):
    """
    Tells what type of file is being used for the data.

    .. attribute:: YAML
        A YAML file is being used.

    .. attribute:: JSON
        A JSON file is being used.

    """

    YAML = 0
    JSON = 1


@dataclass
class LayerParameters:
    """
    Contains parameters needed to instantiate a layer within the PV-T panel.

    .. attribute:: mass
        The mass of the layer, measured in Kelvin.

    .. attribute:: heat_capacity
        The heat capacity of the layer, measured in Joules per kilogram Kelvin.

    .. attribute:: area
        The area of the layer, measured in meters squared.

    .. attribute:: thickness
        The thickness of the layer, measured in meters.

    .. attribute:: temperature
        The temperature at which to initialise the layer, measured in Kelvin.

    """

    mass: float
    heat_capacity: float
    area: float
    thickness: float
    temperature: Optional[float]


@dataclass
class BackLayerParameters(LayerParameters):
    """
    Contains parameters needed to instantiate the back layer of the PV-T panel.

    .. attribute:: conductance
        The conductance of layer (to the environment/its surroundings), measured in
        Watts per meter squared Kelvin.

    """

    conductivity: float


@dataclass
class OpticalLayerParameters(LayerParameters):
    """
    Contains parameters needed to instantiate a layer with optical properties.

    .. attribute:: transmissivity
        The transmissivity of the layer: a dimensionless number between 0 (nothing is
        transmitted through the layer) and 1 (all light is transmitted).

    .. attribute:: absorptivity
        The absorptivity of the layer: a dimensionless number between 0 (nothing is
        absorbed by the layer) and 1 (all light is absorbed).

    .. attribute:: emissivity
        The emissivity of the layer; a dimensionless number between 0 (nothing is
        emitted by the layer) and 1 (the layer re-emits all incident light).

    """

    transmissivity: float
    absorptivity: float
    emissivity: float


@dataclass
class CollectorParameters(OpticalLayerParameters):
    """
    Contains parameters needed to instantiate a collector layer within the PV-T panel.

    .. attribute:: length
        The legnth of the collector, measured in meters.

    .. attribute:: number_of_pipes
        The number of pipes attached to the back of the thermal collector.
        NOTE: This parameter is very geography/design-specific, and will only be
        relevant/useful to the current design of collector being modeled. Namely, when
        multiple pipes flow linearly down the length of the collector, with the HTF
        taking a single pass through the collector.

    .. attribute:: output_water_temperature
        The temperature, in Kelvin, of water outputted by the layer.

    .. attribute:: pipe_diameter
        The diameter of the pipe, in meters.

    .. attribute:: mass_flow_rate
        The mass flow rate of heat-transfer fluid through the collector.

    .. attribute:: htf_heat_capacity
        The heat capacity of the heat-transfer fluid through the collector, measured in
        Joules per kilogram Kelvin.

    .. attribute:: pump_power
        The electrical power, in Watts, consumed by the water pump in the collector.

    """

    length: float
    number_of_pipes: float
    output_water_temperature: float
    pipe_diameter: float
    mass_flow_rate: float
    htf_heat_capacity: float
    pump_power: float


@dataclass
class PVParameters(OpticalLayerParameters):
    """
    Contains parameters needed to instantiate a PV layer within the PV-T panel.

    .. attribute:: reference_efficiency
        The efficiency of the PV layer at the reference temperature. Thie value varies
        between 1 (corresponding to 100% efficiency), and 0 (corresponding to 0%
        efficiency)

    .. attribute:: reference_temperature
        The referencee temperature, in Kelvin, at which the reference efficiency is
        defined.

    .. attribute:: thermal_coefficient
        The thermal coefficient for the efficiency of the panel.

    """

    reference_efficiency: float
    reference_temperature: float
    thermal_coefficient: float


@dataclass
class WeatherConditions:
    """
    Contains information about the various weather conditions at any given time.

    .. attribute:: irradiance
        The solar irradiance in Watts per meter squared.

    .. attribute:: declination
        The angle of declination of the sun above the horizon

    .. attribute:: azimuthal_angle
        The azimuthal angle of the sun, defined clockwise from True North.

    .. attribute:: wind_speed
        The wind speed in meters per second.

    .. attribute:: ambient_temperature
        The ambient temperature in

    """

    _irradiance: float
    declination: float
    azimuthal_angle: float
    wind_speed: float
    ambient_temperature: float

    @property
    def irradiance(self) -> float:
        """
        The irradiance should only be definied if the sun is above the horizon.

        :return:
            The solar irradiance, adjusted for the day-night cycle.

        """

        if self.declination > 0:
            return self._irradiance
        return 0

    @property
    def sky_temperature(self) -> float:
        """
        Determines the radiative temperature of the sky.

        The "sky," as a black body, has a radiative temperature different to that of the
        surrounding air, or the ambient temperature. This function converts between them
        and outputs the sky's radiative temperature.

        :return:
            The radiative temperature of the "sky" in Kelvin.

        """

        return 0.0552 * (self.ambient_temperature ** 1.5)

    def __repr__(self) -> str:
        """
        Return a nice representation of the weather conditions.

        :return:
            A nicely-formatted string containing weather conditions data.

        """

        return (
            "WeatherConditions(irradiance: {}, declination: {}, ".format(
                self.irradiance, self.declination
            )
            + "azimuthal_angle: {}, wind_speed: {}, ambient_temperature: {}, ".format(
                self.azimuthal_angle, self.wind_speed, self.ambient_temperature
            )
            + "sky_temperature: {}".format(self.sky_temperature)
        )


def read_yaml(yaml_file_path: str) -> Dict[Any, Any]:
    """
    Read in some yaml data and return it.

    :param yaml_file_path:
        The path to the yaml data to read in.

    :return:
        A `dict` containing the data read in from the yaml file.

    """

    logger = logging.getLogger(LOGGER_NAME)

    # Open the yaml data and read it.
    if not os.path.isfile(yaml_file_path):
        logger.error(
            "A YAML data file, '%s', could not be found. Exiting...", yaml_file_path
        )
        raise FileNotFoundError(yaml_file_path)
    with open(yaml_file_path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.parser.ParserError as e:
            logger.error("Failed to read YAML file '%s'.", yaml_file_path)
            print(f"Failed to parse YAML. Internal error: {str(e)}")
            raise

    logger.info("Data successfully read from '%s'.", yaml_file_path)
    return data


def get_logger(logger_name: str) -> logging.Logger:
    """
    Set-up and return a logger.

    :param logger_name:
        The name of the logger to instantiate.

    :return:
        The logger for the component.

    """

    # Create a logger with the current component name.
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Create a file handler which logs even debug messages.
    if os.path.exists(f"{logger_name}.log"):
        os.rename(f"{logger_name}.log", f"{logger_name}.log.1")
    fh = logging.FileHandler(f"{logger_name}.log")
    fh.setLevel(logging.DEBUG)
    # Create a console handler with a higher log level.
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # Create a formatter and add it to the handlers.
    formatter = logging.Formatter(
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
