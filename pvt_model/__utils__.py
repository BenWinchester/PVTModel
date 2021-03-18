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
from typing import Dict, Optional

__all__ = (
    "BColours",
    "CarbonEmissions",
    "COARSE_RUN_RESOLUTION",
    "fourier_number",
    "get_logger",
    "INITIAL_CONDITION_PRECISION",
    "LOGGER_NAME",
    "ProgrammerJudgementFault",
    "MissingParametersError",
    "SystemData",
    "TotalPowerData",
)

# The resolution in seconds for determining the initial conditions.
COARSE_RUN_RESOLUTION: int = 1800
# The prevision to reach when searching for consistent initial temperatures.
INITIAL_CONDITION_PRECISION: float = 1
# The directory for storing the logs.
LOGGER_DIRECTORY = "logs"
# The name used for the internal logger.
LOGGER_NAME = "pvt_model"


@dataclass
class BColours:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class CarbonEmissions:
    """
    Contains information about the carbon emissions produced.

    .. attribute:: electrical_carbon_produced
        The amount of CO2 equivalent produced, measured in kilograms.

    .. attribute:: electrical_carbon_saved
        The amount of CO2 saved by using PV-T, measured in kilograms.

    .. attribute:: heating_carbon_produced
        The amount of CO2 equivalent produced, measured in kilograms.

    .. attribute:: heating_carbon_saved
        The amount of CO2 equivalent saved by using the PV-T, measured in kilograms.

    """

    electrical_carbon_produced: float
    electrical_carbon_saved: float
    heating_carbon_produced: float
    heating_carbon_saved: float


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


def fourier_number(
    conductive_length_scale: float,
    conductivity: float,
    density: float,
    heat_capacity: float,
    time_scale: float,
) -> float:
    """
    Calculates the Fourier coefficients based off the parameters passed in.

    The Fourier number is computed by:
        fourier_number = (
            conductivity * time_scale
        ) / (
            conductivy_length_scale ^ 2 * density * heat_capacity
        )

    :param conductive_length_scale:
        The length scale over which conduction occurs, measured in meters.

    :param conductivity:
        The thermal conductivity of the material, measured in Watts per meter Kelvin.

    :param density:
        The density of the medium, measured in kilograms per meter cubed.

    :param heat_capacity:
        The heat capcity of the conductive medium, measured in Joules per kilogram
        Kelvin.

    :param time_scale:
        The time scale of the simulation being run, measured in seconds.

    :return:
        The Fourier number based on these values.

    """

    f_num: float = (conductivity * time_scale) / (  # [W/m*K] * [s]
        heat_capacity  # [J/kg*K]
        * density  # [kg/m^3]
        * conductive_length_scale ** 2  # [m]^2
    )

    return f_num


def get_logger(logger_name: str, verbose: bool) -> logging.Logger:
    """
    Set-up and return a logger.

    :param logger_name:
        The name of the logger to instantiate.

    :param verbose:
        Whether the logging is verbose (DEBUG reported) or not (INFO only).

    :return:
        The logger for the component.

    """

    # Create a logger with the current component name.
    logger = logging.getLogger(logger_name)
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # Create the logging directory if it doesn't exist.
    if not os.path.isdir(LOGGER_DIRECTORY):
        os.mkdir(LOGGER_DIRECTORY)
    # Create a file handler which logs even debug messages.
    if os.path.exists(os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log")):
        os.rename(
            os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log"),
            os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log.1"),
        )
    fh = logging.FileHandler(os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log"))
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


@dataclass
class SystemData:
    """
    Contains information about the system at a given time step.

    .. attribute:: ambient_temperature
        The ambient temperature, measured in Celcius.

    .. attribute:: bulk_water_temperature
        The temperature of the bulk water, measured in Celcius.

    .. attribute:: collector_temperature
        The temperature of the collector layer, measured in Celcius.

    .. attribute:: date
        A `str` giving the current date.

    .. attribute:: exchanger_temperature_drop
        The temperature drop through the heat exchanger in the tank, measured in Kelvin
        or Celcius. (As it is a temperature difference, the two scales are equivalent.)

    .. attribute:: glass_temperature
        The temperature of the glass layer, measured in Celcius.

    .. attribute:: pipe_temperature
        The temperature of the pipe, measured in Celcius.

    .. attribute:: pv_temperature
        The temperature of the PV layer, measured in Celcius.

    .. attribute:: sky_temperature
        The temperature of the sky, measured in Celcius.

    .. attribute:: tank_temperature
        The temperature of the hot-water tank, measured in Celcius.

    .. attribute:: time
        A `str` giving the current time.

    .. attribute:: layer_temperature_map_bulk_water
        A mapping between coordinate and temperature for the bulk water within the
        pipes.

    .. attribute:: layer_temperature_map_collector
        A mapping between coordinate and temperature for segments within the collector
        layer.

    .. attribute:: layer_temperature_map_glass
        A mapping between coordinate and temperature for segments within the glass layer.

    .. attribute:: layer_temperature_map_pipe
        A mapping between coordinate and temperature for the pipes.

    .. attribute:: layer_temperature_map_pv
        A mapping between coordinate and temperature for segments within the pv layer.

    .. attribute:: collector_input_temperature
        The temperature of the HTF inputted into the collector, measured in Celcius.
        This can be set to `None` if no data is recorded.

    .. attribute:: collector_output_temperature
        The temperature of the HTF outputted from the collector, measured in Celcius.
        This can be set to `None` if no data is recorded.

    """

    ambient_temperature: float
    bulk_water_temperature: float
    collector_temperature: float
    date: str
    glass_temperature: float
    exchanger_temperature_drop: float
    pipe_temperature: float
    pv_temperature: float
    sky_temperature: float
    tank_temperature: float
    time: str
    collector_input_temperature: Optional[float] = None
    collector_output_temperature: Optional[float] = None
    layer_temperature_map_bulk_water: Optional[Dict[str, float]] = None
    layer_temperature_map_collector: Optional[Dict[str, float]] = None
    layer_temperature_map_glass: Optional[Dict[str, float]] = None
    layer_temperature_map_pipe: Optional[Dict[str, float]] = None
    layer_temperature_map_pv: Optional[Dict[str, float]] = None


class TemperatureName(enum.Enum):
    """
    Used for keeping track of the temperature value being used.

    """

    glass = 0
    pv = 1
    collector = 2
    pipe = 3
    htf = 4
    htf_in = 5
    htf_out = 6
    collector_in = 7
    collector_out = 8
    tank_in = 9
    tank_out = 10
    tank = 11


@dataclass
class TotalPowerData:
    """
    Contains information about the total power generated by the system.

    .. attribute:: electricity_supplied
        The electricity supplied, measured in Joules.

    .. attribute:: electricity_demand
        The electricity demand, measured in Joules.

    .. attribute:: heating_supplied
        The heatimg supplied, measured in Joules.

    .. attribute:: heating_demand
        The heating demand, measured in Joules.

    """

    electricity_supplied: float = 0
    electricity_demand: float = 0
    heating_supplied: float = 0
    heating_demand: float = 0

    def increment(
        self,
        electricity_supplied_incriment,
        electricity_demand_incriment,
        heating_supplied_increment,
        heating_demand_increment,
    ) -> None:
        """
        Updates the values by incrementing with those supplied.

        :param electricity_supplied_increment:
            Electricity supplied to add, measured in Joules.
        :param electricity_demand_increment:
            Electricity demand to add, measured in Joules.
        :param heating_supplied_increment:
            Heating supplied to add, measured in Joules.
        :param heating_demand_increment:
            Heating demand to add, measured in Joules.

        """

        self.electricity_supplied += electricity_supplied_incriment
        self.electricity_demand += electricity_demand_incriment
        self.heating_supplied += heating_supplied_increment
        self.heating_demand += heating_demand_increment
