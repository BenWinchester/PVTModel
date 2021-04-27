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
    "BColours",
    "CarbonEmissions",
    "COARSE_RUN_RESOLUTION",
    "fourier_number",
    "get_logger",
    "INITIAL_CONDITION_PRECISION",
    "InvalidParametersError",
    "LOGGER_NAME",
    "OperatingMode",
    "ProgrammerJudgementFault",
    "MissingParametersError",
    "read_yaml",
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
    """
    Contains various colours used for pretty-printing out to the command-line on stdout.

    .. attribute:: FAIL
        Used for a failure message.

    .. attributes:: OKTEAL, WARNING, OKBLUE, HEADER, OKCYAN, OKGREEN
        Various colours used.

    .. attribute:: ENDC
        Used to reset the colour of the terminal output.

    .. attribute:: BOLD, UNDERLINE
        Used to format the text.

    """

    FAIL = "\033[91m"
    OKTEAL = "\033[92m"
    WARNING = "\033[93m"
    OKBLUE = "\033[94m"
    HEADER = "\033[95m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[97m"
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
    # Create the logging directory if it doesn't exist.
    if not os.path.isdir(LOGGER_DIRECTORY):
        os.mkdir(LOGGER_DIRECTORY)
    # Rename old log files.
    append_index = 1
    if os.path.exists(os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log")):
        while os.path.exists(os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log.{append_index}")):
            append_index += 1
    fh = logging.FileHandler(os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log.{append_index}"))
    ch = logging.StreamHandler()
    if verbose:
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.WARN)
    else:
        logger.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
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


class InvalidParametersError(Exception):
    """
    Raised when some parameters have been specified incorrectly.

    """

    def __init__(self, message: str, variable_name: str) -> None:
        """
        Instantiate an invalid parameters error.

        :param message:
            An appended message to display to the user.

        :param variable_name:
            The name of the variable for which insufficient parameters were specified.

        """

        super().__init__(
            f"Invalid parameters when determining '{variable_name}': {message}."
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


@dataclass
class OperatingMode:
    """
    Contains information about the mode of operation of the model.

    .. attribute:: coupled
        Whether the system is coupled (True) or decoupled (False)

    .. attribute:: dynamic
        Whether the system is dyanmic (True) or steady-state (False).

    """

    coupled: bool
    dynamic: bool

    @property
    def decoupled(self) -> bool:
        """
        Returns whether the operating mode is decoupled.

        :return:
            A `bool` giving whether the system is coupled (False) or decoupled (True).

        """

        return not self.coupled

    @property
    def steady_state(self) -> bool:
        """
        Returns whether the operating mode is steady-state.

        :return:
            A `bool` giving whether the system is dynamic (False) or steady-state
            (True).

        """

        return not self.dynamic


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
            data: Dict[Any, Any] = yaml.safe_load(f)
        except yaml.parser.ParserError as e:
            logger.error("Failed to read YAML file '%s'.", yaml_file_path)
            print(f"Failed to parse YAML. Internal error: {str(e)}")
            raise

    logger.info("Data successfully read from '%s'.", yaml_file_path)
    return data


@dataclass
class SystemData:
    """
    Contains information about the system at a given time step.

    .. attribute:: ambient_temperature
        The ambient temperature, measured in Celcius.

    .. attribute:: bulk_water_temperature
        The temperature of the bulk water, measured in Celcius.

    .. attribute:: absorber_temperature
        The temperature of the absorber layer, measured in Celcius.

    .. attribute:: date
        A `str` giving the current date.

    .. attribute:: electrical_efficiency
        The electrical efficiency of the PV layer of the collector.

    .. attribute:: exchanger_temperature_drop
        The temperature drop through the heat exchanger in the tank, measured in Kelvin
        or Celcius. (As it is a temperature difference, the two scales are equivalent.)

    .. attribute:: glass_temperature
        The temperature of the glass layer, measured in Celcius.  Can be `None` if no
        glass present.

    .. attribute:: pipe_temperature
        The temperature of the pipe, measured in Celcius.

    .. attribute:: pv_temperature
        The temperature of the PV layer, measured in Celcius. Can be `None` if no PV
        present.

    .. attribute:: reduced_temperature
        The reduced temperature of the PV-T absorber, measured in Celcius.

    .. attribute:: sky_temperature
        The temperature of the sky, measured in Celcius.

    .. attribute:: solar_irradiance
        The solar irradiance incident on the PV-T collector, measured in Watts per meter
        squared.

    .. attribute:: tank_temperature
        The temperature of the hot-water tank, measured in Celcius.

    .. attribute:: thermal_efficiency
        The thermal efficiency of the system.

    .. attribute:: upper_glass_temperature
        The temperature of the upper-glass (i.e., double-glazing) temperature, measured
        in Celcius. Can be `None` if no double-glazing present.

    .. attribute:: collector_input_temperature
        The temperature of the HTF inputted into the absorber, measured in Celcius.
        This can be set to `None` if no data is recorded.

    .. attribute:: collector_output_temperature
        The temperature of the HTF outputted from the absorber, measured in Celcius.
        This can be set to `None` if no data is recorded.

    .. attribute:: collector_temperature_gain
        The temperature gain of the HTF through the thermal collector.

    .. attribute:: layer_temperature_map_bulk_water
        A mapping between coordinate and temperature for the bulk water within the
        pipes.

    .. attribute:: layer_temperature_map_absorber
        A mapping between coordinate and temperature for elements within the absorber
        layer.

    .. attribute:: layer_temperature_map_glass
        A mapping between coordinate and temperature for elements within the glass layer.

    .. attribute:: layer_temperature_map_pipe
        A mapping between coordinate and temperature for the pipes.

    .. attribute:: layer_temperature_map_pv
        A mapping between coordinate and temperature for elements within the pv layer.

    .. attribute:: layer_temperature_map_upper_glass
        A mapping between coordinate and temperature for elements within the upper-glass
        (i.e., double-glazing) layer.

    .. attribute:: reduced_collector_temperature
        The reduced temperature of the collector.

    .. attribute:: time
        A `str` giving the current time, can be set to `None` for steady-state runs.

    """

    absorber_temperature: float
    ambient_temperature: float
    bulk_water_temperature: float
    date: str
    electrical_efficiency: Optional[float]
    glass_temperature: Optional[float]
    exchanger_temperature_drop: Optional[float]
    pipe_temperature: float
    pv_temperature: float
    reduced_collector_temperature: Optional[float]
    sky_temperature: float
    solar_irradiance: float
    tank_temperature: Optional[float]
    thermal_efficiency: Optional[float]
    upper_glass_temperature: Optional[float]
    collector_input_temperature: Optional[float] = None
    collector_output_temperature: Optional[float] = None
    collector_temperature_gain: Optional[float] = None
    layer_temperature_map_bulk_water: Optional[Dict[str, float]] = None
    layer_temperature_map_absorber: Optional[Dict[str, float]] = None
    layer_temperature_map_glass: Optional[Dict[str, float]] = None
    layer_temperature_map_pipe: Optional[Dict[str, float]] = None
    layer_temperature_map_pv: Optional[Dict[str, float]] = None
    layer_temperature_map_upper_glass: Optional[Dict[str, float]] = None
    reduced_temperature: Optional[float] = None
    time: Optional[str] = None


class TemperatureName(enum.Enum):
    """
    Used for keeping track of the temperature value being used.

    """

    upper_glass = 0
    glass = 1
    pv = 2
    absorber = 3
    pipe = 4
    htf = 5
    htf_in = 6
    htf_out = 7
    collector_in = 8
    collector_out = 9
    tank_in = 10
    tank_out = 11
    tank = 12


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
