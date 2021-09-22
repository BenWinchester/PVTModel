#!/usr/bin/python3.7
########################################################################################
# __main__.py - The main module for this, my first, PV-T model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The main module for the PV-T model.

This module coordinates the time-stepping of the itterative module, calling out to the
various modules where necessary, as well as reading in the various data files and
processing the command-line arguments that define the scope of the model run.

"""

import logging
import os

from typing import Dict, List, Optional, Set, Tuple

import numpy


from . import (
    exchanger,
    index_handler,
    load,
    mains_power,
    process_pvt_system_data,
    tank,
    weather,
)

from .runs import coupled, decoupled

from ..__utils__ import (  # pylint: disable=unused-import
    BColours,
    CarbonEmissions,
    get_logger,
    OperatingMode,
    SystemData,
    TemperatureName,
    TotalPowerData,
)
from .__utils__ import (
    ProgrammerJudgementFault,
    PVT_SYSTEM_MODEL_LOGGER_NAME,
    SOLAR_IRRADIANCE_FOLDERNAME,
    TEMPERATURE_FOLDERNAME,
    WEATHER_DATA_FILENAME,
)

from .constants import (
    DEFAULT_INITIAL_DATE_AND_TIME,
    ZERO_CELCIUS_OFFSET,
)


def _get_load_system(location: str) -> load.LoadSystem:
    """
    Instantiates a :class:`load.LoadSystem` instance based on the file data.

    :param locataion:
        The location being considered for which to instantiate the load system.

    :return:
        An instantiated :class:`load.LoadSystem` based on the input location.

    """

    load_profiles = {
        os.path.join(location, "load_profiles", filename)
        for filename in os.listdir(os.path.join(location, "load_profiles"))
    }
    load_system: load.LoadSystem = load.LoadSystem.from_data(load_profiles)
    return load_system


def _get_weather_forecaster(
    average_irradiance: bool,
    location: str,
    use_pvgis: bool,
    override_ambient_temperature: Optional[float],
    override_irradiance: Optional[float],
    override_wind_speed: Optional[float],
) -> weather.WeatherForecaster:
    """
    Instantiates a :class:`weather.WeatherForecaster` instance based on the file data.

    :param average_irradiance:
        Whether to use an average irradiance profile for the month (True) or use
        irradiance profiles for each day individually (False).

    :param location:
        The location currently being considered, and for which to instantiate the
        :class:`weather.WeatherForecaster` instance.

    :param override_ambient_temperature:
        Overrides the ambient temperature value. The value should be in degrees Celcius.

    :param override_irradiance:
        Overrides the irradiance value. The value should be in Watts per meter squared.

    :param override_wind_speed:
        Overrides the wind speed value. The value should be in meters per second.

    :param use_pvgis:
        Whether data from the PVGIS (Photovoltaic Geographic Information Survey) should
        be used (True) or not (False).

    :return:
        An instantiated :class:`weather.WeatherForecaster` instance based on the
        supplied parameters.

    """

    solar_irradiance_filenames = {
        os.path.join(location, SOLAR_IRRADIANCE_FOLDERNAME, filename)
        for filename in os.listdir(os.path.join(location, SOLAR_IRRADIANCE_FOLDERNAME))
    }
    temperature_filenames = {
        os.path.join(location, TEMPERATURE_FOLDERNAME, filename)
        for filename in os.listdir(os.path.join(location, TEMPERATURE_FOLDERNAME))
    }

    weather_forecaster: weather.WeatherForecaster = weather.WeatherForecaster.from_data(
        average_irradiance,
        solar_irradiance_filenames,
        temperature_filenames,
        os.path.join(location, WEATHER_DATA_FILENAME),
        override_ambient_temperature,
        override_irradiance,
        override_wind_speed,
        use_pvgis,
    )
    return weather_forecaster


######################
# System run methods #
######################


def main(  # pylint: disable=too-many-branches
    average_irradiance: bool,
    cloud_efficacy_factor: float,
    disable_logging: bool,
    exchanger_data_file: str,
    initial_month: int,
    initial_system_temperature_vector: List[float],
    layers: Set[TemperatureName],
    location: str,
    operating_mode: OperatingMode,
    portion_covered: float,
    pvt_data_file: str,
    resolution: int,
    save_2d_output: bool,
    tank_data_file: str,
    use_pvgis: bool,
    verbose: bool,
    x_resolution: int,
    y_resolution: int,
    *,
    run_number: Optional[int],
    start_time: int,
    override_ambient_temperature: Optional[float] = None,
    override_collector_input_temperature: Optional[float] = None,
    override_irradiance: Optional[float] = None,
    override_mass_flow_rate: Optional[float] = None,
    override_wind_speed: Optional[float] = None,
    days: Optional[int] = None,
    minutes: Optional[int] = None,
    months: Optional[int] = None,
) -> Tuple[numpy.ndarray, Dict[float, SystemData]]:
    """
    The main module for the code. Calling this method executes a run of the simulation.

    :param average_irradiance:
        Whether to use an average irradiance profile for the month.

    :param cloud_efficiacy_factor:
        The extent to which cloud cover influences the solar irradiance.

    :param disable_logging:
        Whether to disable the file handler for the logging (True) or keep it enabled
        (False).

    :param exchanger_data_file:
        The path to the data file containing information about the heat exchanger being
        modelled in the run.

    :param initial_month:
        The initial month for the run, given between 1 (January) and 12 (December).

    :param initial_system_temperature_vector:
        The vector of initial temperatures used to model the system.

    :param location:
        The location at which the PVT system is installed and for which location data
        should be used.

    :param operating_mode:
        The operating mode of the run, containing information needed to set up the
        matrix.

    :param portion_covered:
        The portion of the absorber which is covered with PV cells.

    :param pvt_data_file:
        The path to the data file containing information about the PVT system being
        modelled.

    :param resolution:
        The temporal resolution at which to run the simulation.

    :param save_2d_output:
        If True, the 2D output is saved to the system data and returned. If False, only
        the 1D output is saved.

    :param tank_data_file:
        The path to the data file containing information about the hot-water tank used
        in the model.

    :param unglazed:
        If specified, the glass cover is not included in the model.

    :param use_pvgis:
        Whether the data obtained from the PVGIS system should be used.

    :param verbose:
        Whether the logging level is verbose (True) or not (False).

    :param x_resolution:
        The x resolution of the simulation being run.

    :param y_resolution:
        The y resolution of the simulation being run.

    :param run_number:
        The number of the run being carried out. This is used for categorising logs.

    :param start_time:
        The time of day at which to start the simulation, specified between 0 and 23.
        This can be `None` if a steady-state simulation is being run.

    :param override_ambient_temperature:
        In decoupled instances, the ambient temperature can be specified as a constant
        value which will override the ambient-temperature profiles.

    :param override_collector_input_temperature:
        In decoupled instances, the collector input temperature can be specified as a
        constant value which will override the dynamic behaviour. This should be
        specified in Kelvin.

    :param override_irradiance:
        In decoupled instances, the solar irradiance can be specified as a constant
        value which will override the solar-irradiance profiles.

    :param override_mass_flow_rate:
        If provided, this will override the mass-flow rate used in the collector.

    :param override_wind_speed:
        In decoupled instances, the wind speed can be specified as a cosntant value
        which will override the wind-speed profiles.

    :param days:
        The number of days for which the simulation is being run. This can be `None` if
        a steady-state simulation is being run.

    :param minutes:
        The number of minutes for which the simulation is being run. This can be `None`
        if either a steady-state simulation is being run, or if either days or months
        have been specified.

    :param months:
        The number of months for which to run the simulation. This can be `None` if a
        steady-state simulation is being run.

    :return:
        The system data is returned.

    """

    # Get the logger for the component.
    if operating_mode.dynamic:
        logger = get_logger(
            disable_logging,
            PVT_SYSTEM_MODEL_LOGGER_NAME.format(
                tag=f"{resolution}s", run_number=run_number
            ),
            verbose,
        )
    else:
        logger = get_logger(
            disable_logging,
            PVT_SYSTEM_MODEL_LOGGER_NAME.format(
                tag="steady_state", run_number=run_number
            ),
            verbose,
        )

    # Set up numpy printing style.
    numpy.set_printoptions(formatter={"float": "{: 0.3f}".format})

    # Set up the weather module.
    weather_forecaster = _get_weather_forecaster(
        average_irradiance,
        location,
        use_pvgis,
        override_ambient_temperature,
        override_irradiance,
        override_wind_speed,
    )
    logger.info("Weather forecaster successfully instantiated: %s", weather_forecaster)

    # Set up the load module.
    if operating_mode.coupled:
        load_system: Optional[load.LoadSystem] = _get_load_system(location)
        logger.info(
            "Load system successfully instantiated: %s",
            load_system,
        )
    else:
        load_system = None

    # Raise an exception if no initial system temperature vector was supplied.
    if initial_system_temperature_vector is None:
        raise ProgrammerJudgementFault(
            "Not initial system temperature vector was supplied. This is necessary "
            "when calling the pvt model."
        )

    # Initialise the PV-T panel.
    pvt_collector = process_pvt_system_data.pvt_collector_from_path(
        layers,
        logger,
        override_mass_flow_rate,
        portion_covered,
        pvt_data_file,
        x_resolution,
        y_resolution,
    )
    logger.info("PV-T panel successfully instantiated: %s", pvt_collector)
    logger.debug(
        "PV-T panel elements:\n  %s",
        "\n  ".join(
            [
                f"{element_coordinates}: {element}"
                for element_coordinates, element in pvt_collector.elements.items()
            ]
        ),
    )

    # Instantiate the rest of the PVT system if relevant.
    if operating_mode.coupled:
        # Set up the heat exchanger.
        heat_exchanger: Optional[
            exchanger.Exchanger
        ] = process_pvt_system_data.heat_exchanger_from_path(exchanger_data_file)
        logger.info("Heat exchanger successfully instantiated: %s", heat_exchanger)

        # Set up the hot-water tank.
        hot_water_tank: Optional[
            tank.Tank
        ] = process_pvt_system_data.hot_water_tank_from_path(tank_data_file)
        logger.info("Hot-water tank successfully instantiated: %s", hot_water_tank)

        # Set up the mains supply system.
        mains_supply = mains_power.MainsSupply.from_yaml(
            os.path.join(location, "utilities.yaml")
        )
        logger.info("Mains supply successfully instantiated: %s", mains_supply)

    else:
        heat_exchanger = None
        logger.info("NO HEAT EXCHANGER PRESENT")
        hot_water_tank = None
        logger.info("NO HOT-WATER TANK PRESENT")
        mains_supply = None
        logger.info("NO MAINS SUPPLY PRESENT")

    # Determine the number of temperatures being modelled.
    number_of_pipes = len(
        {
            element.pipe_index
            for element in pvt_collector.elements.values()
            if element.pipe_index is not None
        }
    )
    number_of_x_elements = len(
        {element.x_index for element in pvt_collector.elements.values()}
    )
    number_of_y_elements = len(
        {element.y_index for element in pvt_collector.elements.values()}
    )
    if operating_mode.coupled:
        number_of_temperatures: int = index_handler.num_temperatures(pvt_collector)
    else:
        number_of_temperatures = index_handler.num_temperatures(pvt_collector) - 3
    logger.info(
        "System consists of %s pipes, %s by %s elements, and %s temperatures in all.",
        number_of_pipes,
        number_of_x_elements,
        number_of_y_elements,
        number_of_temperatures,
    )

    # Instantiate the two pipes used to store input and output temperature values.
    # absorber_to_tank_pipe = pipe.Pipe(temperature=initial_system_temperature_vector[3])
    # tank_to_absorber_pipe = pipe.Pipe(temperature=initial_system_temperature_vector[3])

    # Instnatiate the hot-water pump.
    # htf_pump = process_pvt_system_data.pump_from_path(pump_data_file)

    # Set up a holder for the information about the final output of the system.
    # total_power_data = TotalPowerData()

    logger.info(
        "System state before beginning run:\n%s\n%s\n%s\n%s",
        heat_exchanger if heat_exchanger is not None else "No heat exchanger",
        hot_water_tank if hot_water_tank is not None else "No hot-water tank",
        pvt_collector,
        weather_forecaster,
    )

    if operating_mode.coupled and operating_mode.dynamic:
        if heat_exchanger is None or hot_water_tank is None or load_system is None:
            raise ProgrammerJudgementFault(
                "{}{} not defined in dynamic operation.{}".format(
                    BColours.FAIL,
                    ", ".join(
                        {
                            entry
                            for entry in {
                                "heat exchanger" if heat_exchanger is None else "",
                                "hot-water tank" if hot_water_tank is None else "",
                                "load system" if load_system is None else "",
                            }
                            if entry is not None
                        }
                    ),
                    BColours.ENDC,
                )
            )
        final_run_temperature_vector, system_data = coupled.coupled_dynamic_run(
            cloud_efficacy_factor,
            days,
            heat_exchanger,
            hot_water_tank,
            initial_month,
            initial_system_temperature_vector,
            load_system,
            logger,
            minutes,
            months,
            number_of_pipes,
            number_of_temperatures,
            number_of_x_elements,
            number_of_y_elements,
            operating_mode,
            pvt_collector,
            resolution,
            save_2d_output,
            start_time,
            weather_forecaster,
        )
    elif operating_mode.decoupled and operating_mode.steady_state:
        if override_collector_input_temperature is None:
            raise ProgrammerJudgementFault(
                "{}Override collector input temperature not provided.{}".format(
                    BColours.FAIL, BColours.ENDC
                )
            )
        (
            final_run_temperature_vector,
            system_data_entry,
        ) = decoupled.decoupled_steady_state_run(
            override_collector_input_temperature,
            cloud_efficacy_factor,
            DEFAULT_INITIAL_DATE_AND_TIME.replace(month=initial_month),
            initial_system_temperature_vector,
            logger,
            number_of_pipes,
            number_of_temperatures,
            number_of_x_elements,
            number_of_y_elements,
            operating_mode,
            pvt_collector,
            save_2d_output,
            weather_forecaster,
        )
        system_data = {
            override_collector_input_temperature
            - ZERO_CELCIUS_OFFSET: system_data_entry[1]
        }
    elif operating_mode.decoupled and operating_mode.dynamic:
        if override_collector_input_temperature is None:
            raise ProgrammerJudgementFault(
                "{}Override collector input temperature not provided.{}".format(
                    BColours.FAIL, BColours.ENDC
                )
            )
        (final_run_temperature_vector, system_data,) = decoupled.decoupled_dynamic_run(
            cloud_efficacy_factor,
            override_collector_input_temperature,
            days,
            initial_month,
            initial_system_temperature_vector,
            logger,
            minutes,
            months,
            number_of_pipes,
            number_of_temperatures,
            number_of_x_elements,
            number_of_y_elements,
            operating_mode,
            pvt_collector,
            resolution,
            save_2d_output,
            start_time,
            weather_forecaster,
        )
    else:
        raise ProgrammerJudgementFault(
            "The system model was called with an invalid operating mode: {}.".format(
                operating_mode
            )
        )

    return final_run_temperature_vector, system_data


if __name__ == "__main__":
    logging.error(
        "Calling the internal `pvt_system` from the command-line is no longer "
        "supported."
    )
    raise ProgrammerJudgementFault(
        "Calling the internal `pvt_system` from the command-line interface is no "
        "longer supported."
    )
