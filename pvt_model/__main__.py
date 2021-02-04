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

import dataclasses
import datetime
import math
import os
import sys

from typing import Dict, Optional

import json
import yaml

from dateutil.relativedelta import relativedelta

from . import (
    argparser,
    efficiency,
    load,
    mains_power,
    process_pvt_system_data,
    pump,
    weather,
)
from .__utils__ import (
    CarbonEmissions,
    FileType,
    get_logger,
    HEAT_CAPACITY_OF_WATER,
    INITIAL_SYSTEM_TEMPERATURE,
    LOGGER_NAME,
    MissingParametersError,
    time_iterator,
    TotalPowerData,
    ZERO_CELCIUS_OFFSET,
)


# The temperature of hot-water required by the end-user, measured in Kelvin.
HOT_WATER_DEMAND_TEMP = 60 + ZERO_CELCIUS_OFFSET
# The initial date and time for the simultion to run from.
INITIAL_DATE_AND_TIME = datetime.datetime(2005, 1, 1, 0, 0)
# The average temperature of the air surrounding the tank, which is internal to the
# household, measured in Kelvin.
INTERNAL_HOUSEHOLD_AMBIENT_TEMPERATURE = ZERO_CELCIUS_OFFSET + 20  # [K]
# Get the logger for the component.
logger = get_logger(LOGGER_NAME)
# Folder containing the solar irradiance profiles
SOLAR_IRRADIANCE_FOLDERNAME = "solar_irradiance_profiles"
# Folder containing the temperature profiles
TEMPERATURE_FOLDERNAME = "temperature_profiles"
# Name of the weather data file.
WEATHER_DATA_FILENAME = "weather.yaml"


@dataclasses.dataclass
class SystemData:
    """
    Contains PVT system data at a given time step.

    .. attribute:: ambient_temperature
        The ambient temperature, in Celcius.

    .. attribute:: auxiliary_heating
        The additional energy needed to be supplied to the system through the auxiliary
        heater when the tank temperature is below the required thermal output
        temperature, measured in Watts.

    .. attribute:: back_plate_heat_loss
        The heat lost from the back plate, measured in Watts.

    .. attribute:: bulk_water_heat_gain
        The heat gained by the bulk water, measured in Watts.

    .. attribute:: bulk_water_temperature
        The temperature of the bulk water within the collector, measured in Celcius.

    .. attribute:: collector_heat_input
        The heat inputted to the collector, from both the sun and upper PV layer if
        present, measured in Watts.

    .. attribute:: collector_input_temperature
        The temperature of water flowing into the collector, measured in Celcius.

    .. attribute:: collector_output_temperature
        The temperature of the HTF outputted from the collector, measured in Celcius.

    .. attribute:: collector_temperature
        The temperature of the thermal collector, measured in Celcius.

    .. attribute:: collector_temperature_gain
        The temperature gain of the HTF through the collector, measured in Celcius.

    .. attribute:: date
        The date, formatted as "DD/MM/YYYY".

    .. attribute:: dc_electrical
        The electrical demand covered, defined between 0 and 1.

    .. attribute:: dc_thermal
        The thermal demand covered, defined between 0 and 1.

    .. attribute:: electrical_load
        The load (demand) placed on the PV-T panel's electrical output, measured in
        Watts.

    .. attribute:: exchanger_temperature_drop
        The temperature drop, in Kelvin, across the heat exchanger.

    .. attribute:: glass_temperature
        The temperatuer of the glass layer of the panel, measured in Celcius.

    .. attribute:: gross_electrical_output
        The electrical power produced by the panel, measured in Watts.

    .. attribute:: hot_water_load
        The hot-water load in litres per hour.

    .. attribute:: htf_pump_on
        Whether the HTF pump is on (True) or off (False).

    .. attribute:: net_electrical_output
        The electrical power produced by the panel which is in excess of the demand
        required by the household. IE: gross - demand. This is measured in Watts.

    .. attribute:: normal_irradiance
        The solar irradiance, in Watts, normal to the panel.

    .. attribute:: pv_efficiency
        The efficiency of the PV panel, defined between 0 and 1. This is set to `None`
        if no PV layer is present.

    .. attribute:: pv_temperature
        The temperature of the PV layer of the panel, measured in Celcius. This is set
        to `None` if no PV layer is present.

    .. attribute:: sky_temperature
        The sky temperature, radiatively, in Celcius.

    .. attribute:: solar_irradiance
        The solar irradiance in Watts per meter squared.

    .. attribute:: time
        The time.

    .. attribute:: tank_heat_addition
        The heat added to the tank, in Watts.

    .. attribute:: tank_output_temperature
        The temperature of the water outputted from the hot-water tank, measured in
        Celcius.

    .. attribute:: tank_temperature
        The temperature of the water within the hot-water tank, measured in Celcius.

    .. attribute:: thermal_load
        The load (demand) placed on the hot-water tank's thermal output, measured in
        Watts.

    .. attribute:: thermal_output
        The thermal output from the PV-T system supplied - this is really a combnination
        of the demand required and the temperature of the system, measured in Watts.

    .. attribute:: upward_collector_heat_loss
        The heat lost, upward, from the thermal collector, measured in Watts.

    .. attribute:: upward_glass_heat_loss
        The heat lost, upawrds, from the glass layer, measured in Watts.

    """

    ambient_temperature: float
    auxiliary_heating: float
    back_plate_heat_loss: float
    bulk_water_heat_gain: float
    bulk_water_temperature: float
    collector_heat_input: float
    collector_input_temperature: float
    collector_output_temperature: float
    collector_temperature: float
    collector_temperature_gain: float
    date: str
    dc_electrical: Optional[float]
    dc_thermal: Optional[float]
    electrical_load: float
    exchanger_temperature_drop: float
    glass_temperature: Optional[float]
    gross_electrical_output: float
    hot_water_load: float
    htf_pump_on: bool
    net_electrical_output: float
    normal_irradiance: float
    pv_temperature: Optional[float]
    pv_efficiency: Optional[float]
    sky_temperature: float
    solar_irradiance: float
    tank_heat_addition: float
    tank_temperature: float
    tank_output_temperature: float
    thermal_load: float
    thermal_output: float
    time: str
    upward_collector_heat_loss: float
    upward_glass_heat_loss: Optional[float]

    def __str__(self) -> str:
        """
        Output a pretty picture.

        :return:
            A `str` containing the system data info.

        """

        return (
            "SystenData("
            f"ambient_temperature: {self.ambient_temperature}K, "
            f"auxiliary_heating: {self.auxiliary_heating}W, "
            f"back_plate_heat_loss: {self.back_plate_heat_loss}W, "
            f"bulk_water_heat_gain: {self.bulk_water_heat_gain}W, "
            f"bulk_water_temperature: {self.bulk_water_temperature}K, "
            f"collector_input_temperature: {self.collector_input_temperature}K, "
            f"collector_output_temperature: {self.collector_output_temperature}K, "
            f"collector_temperature: {self.collector_temperature}K, "
            f"collector_temperature_gain: {self.collector_temperature_gain}K, "
            f"date: {self.date}, "
            f"dc_electrical: {self.dc_electrical}, "
            f"dc_thermal: {self.dc_thermal}, "
            f"electrical_load: {self.electrical_load}W, "
            f"exchanger_temperature_drop: {self.exchanger_temperature_drop}K, "
            f"glass_temperature: {self.glass_temperature}K, "
            f"gross_electrical_output: {self.gross_electrical_output}W, "
            f"htf_pump_on: {self.htf_pump_on}, "
            f"net_electrical_output:{self.net_electrical_output}W, "
            f"normal_irradiance:{self.normal_irradiance}W/m^2, "
            f"pv_temperature:{self.pv_temperature}K, "
            f"pv_efficiency:{self.pv_efficiency}, "
            f"sky_temperature:{self.sky_temperature}K, "
            f"solar_irradiance:{self.solar_irradiance}W/m^2,"
            f"tank_heat_addition:{self.tank_heat_addition}W, "
            f"tank_temperature:{self.tank_temperature}K, "
            f"tank_output_temperature:{self.tank_output_temperature}K, "
            f"thermal_load:{self.thermal_load}W, "
            f"thermal_output:{self.thermal_output}W, "
            f"time:{self.time}, "
            f"upward_collector_heat_loss:{self.upward_collector_heat_loss}W, "
            f"upward_glass_heat_loss:{self.upward_glass_heat_loss}W"
            ")"
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
    average_irradiance: bool, location: str, use_pvgis: bool
) -> weather.WeatherForecaster:
    """
    Instantiates a :class:`weather.WeatherForecaster` instance based on the file data.

    :param average_irradiance:
        Whether to use an average irradiance profile for the month (True) or use
        irradiance profiles for each day individually (False).

    :param location:
        The location currently being considered, and for which to instantiate the
        :class:`weather.WeatherForecaster` instance.

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
        os.path.join(location, WEATHER_DATA_FILENAME),
        solar_irradiance_filenames,
        temperature_filenames,
        use_pvgis,
    )
    return weather_forecaster


def _save_data(
    file_type: FileType,
    output_file_name: str,
    system_data: Dict[int, SystemData],
    carbon_emissions: Optional[CarbonEmissions] = None,
    total_power_data: Optional[TotalPowerData] = None,
) -> None:
    """
    Save data when called. The data entry should be appended to the file.

    :param file_type:
        The file type that's being saved.

    :param output_file_name:
        The destination file name.

    :param system_data:
        The data to save.

    :param carbon_emissions:
        The carbon emissions data for the run.

    :param total_power_data:
        The total power data for the run.

    """

    # Convert the system data entry to JSON-readable format
    system_data_dict = {
        key: dataclasses.asdict(value) for key, value in system_data.items()
    }

    # If we're saving YAML data part-way through, then append to the file.
    if file_type == FileType.YAML:
        with open(f"{output_file_name}.yaml", "a") as output_yaml_file:
            yaml.dump(
                system_data_dict,
                output_yaml_file,
            )

    # If we're dumping JSON, open the file, and append to it.
    if file_type == FileType.JSON:
        # Append the total power and emissions data for the run.
        system_data_dict.update(dataclasses.asdict(total_power_data))  # type: ignore
        system_data_dict.update(dataclasses.asdict(carbon_emissions))  # type: ignore

        # Save the data
        # If this is the initial dump, then create the file.
        if not os.path.isfile(f"{output_file_name}.json"):
            with open(f"{output_file_name}.json", "w") as output_json_file:
                json.dump(
                    system_data_dict,
                    output_json_file,
                    indent=4,
                )
        else:
            with open(f"{output_file_name}.json", "r+") as output_json_file:
                # Read the data and append the current update.
                filedata = json.load(output_json_file)
                filedata.update(system_data_dict)
                # Overwrite the file with the updated data.
                output_json_file.seek(0)
                json.dump(
                    filedata,
                    output_json_file,
                    indent=4,
                )


def main(args) -> None:
    """
    The main module for the code.

    :param args:
        The command-line arguments passed into the component.

    """

    logger.info("Beginning run of PVT model.\nCommand: %s", " ".join(args))

    # Parse the system arguments from the commandline.
    parsed_args = argparser.parse_args(args)

    # Check that the output file is specified, and that it doesn't already exist.
    if parsed_args.output is None or parsed_args.output == "":
        logger.error(
            "An output filename must be provided on the command-line interface."
        )
        raise MissingParametersError(
            "Command-Line Interface", "An output file name must be provided."
        )
    if parsed_args.output.endswith(".yaml") or parsed_args.output.endswith(".json"):
        logger.error("The output filename must be irrespective of data type..")
        raise Exception(
            "The output file must be irrespecitve of file extension/data type."
        )
    if os.path.isfile(f"{parsed_args.output}.yaml"):
        logger.info("The output YAML file specified already exists. Moving...")
        os.rename(f"{parsed_args.output}.yaml", f"{parsed_args.output}.yaml.1")
        logger.info("Output file successfully moved.")
    if os.path.isfile(f"{parsed_args.output}.json"):
        logger.info("The output YAML file specified already exists. Moving...")
        os.rename(f"{parsed_args.output}.json", f"{parsed_args.output}.json.1")
        logger.info("Output file successfully moved.")

    # Set up the weather module.
    weather_forecaster = _get_weather_forecaster(
        parsed_args.average_irradiance, parsed_args.location, parsed_args.use_pvgis
    )
    logger.info("Weather forecaster successfully instantiated: %s", weather_forecaster)

    # Set up the load module.
    load_system = _get_load_system(parsed_args.location)
    logger.info(
        "Load system successfully instantiated: %s",
        load_system,
    )

    # Initialise the PV-T panel.
    pvt_panel = process_pvt_system_data.pvt_panel_from_path(
        INITIAL_SYSTEM_TEMPERATURE,
        parsed_args.portion_covered,
        parsed_args.pvt_data_file,
        parsed_args.unglazed,
    )
    logger.info("PV-T panel successfully instantiated: %s", pvt_panel)

    # Instantiate the rest of the PVT system.
    heat_exchanger = process_pvt_system_data.heat_exchanger_from_path(
        parsed_args.exchanger_data_file
    )
    logger.info("Heat exchanger successfully instantiated: %s", heat_exchanger)
    hot_water_tank = process_pvt_system_data.hot_water_tank_from_path(
        parsed_args.tank_data_file
    )
    logger.info("Hot-water tank successfully instantiated: %s", hot_water_tank)
    htf_pump = process_pvt_system_data.pump_from_path(parsed_args.pump_data_file)

    # Intiailise the mains supply system.
    mains_supply = mains_power.MainsSupply.from_yaml(
        os.path.join(parsed_args.location, "utilities.yaml")
    )
    logger.info("Mains supply successfully instantiated: %s", mains_supply)

    # Loop through all times and iterate the system.
    input_water_temperature: float = (
        parsed_args.input_water_temperature + ZERO_CELCIUS_OFFSET
        if parsed_args.input_water_temperature is not None
        else INITIAL_SYSTEM_TEMPERATURE
    )  # [K]
    num_months = (
        (parsed_args.initial_month if parsed_args.initial_month is not None else 1)
        - 1
        + parsed_args.months
    )  # [months]
    start_month = (
        parsed_args.initial_month
        if 1 <= parsed_args.initial_month <= 12
        else INITIAL_DATE_AND_TIME.month
    )  # [months]
    first_date_and_time = INITIAL_DATE_AND_TIME.replace(
        hour=parsed_args.start_time, month=start_month
    )
    if parsed_args.days is None:
        final_date_and_time = first_date_and_time + relativedelta(
            months=num_months % 12, years=num_months // 12
        )
    else:
        final_date_and_time = first_date_and_time + relativedelta(days=parsed_args.days)

    # Set up a dictionary for storing the system data.
    system_data: Dict[int, SystemData] = dict.fromkeys(
        set(range((final_date_and_time - first_date_and_time).seconds))
    )

    # Set up a holder for the information about the final output of the system.
    total_power_data = TotalPowerData()

    logger.debug(
        "Beginning itterative model:\n  Running from: %s\n  Running to: %s",
        str(first_date_and_time),
        str(final_date_and_time),
    )

    logger.info(
        "System state before beginning run:\n%s\n%s\n%s\n%s",
        heat_exchanger,
        hot_water_tank,
        pvt_panel,
        weather_forecaster,
    )

    for run_number, date_and_time in enumerate(
        time_iterator(
            first_time=first_date_and_time,
            last_time=final_date_and_time,
            internal_resolution=parsed_args.internal_resolution,
            timezone=pvt_panel.timezone,
        )
    ):

        # Generate weather and load conditions from the load and weather classes.
        current_weather = weather_forecaster.get_weather(
            *pvt_panel.coordinates, parsed_args.cloud_efficacy_factor, date_and_time
        )
        current_electrical_load = load_system[
            (load.ProfileType.ELECTRICITY, date_and_time)
        ]  # [Watts]
        current_hot_water_load = (
            load_system[(load.ProfileType.HOT_WATER, date_and_time)]  # [litres/hour]
            * parsed_args.internal_resolution  # [s/time_step]
            / 3600  # [s/hour]
        )  # [litres/time_step] = [kg/time_step]

        # Call the pvt module to generate the new temperatures at this time step.
        # >>> Code in this block runs the PVT updator at a 1 second resolution.
        # pvt_update_output = tuple()
        # for _ in range(parsed_args.internal_resolution):
        #     if len(pvt_update_output) == 0:
        #         pvt_update_output = pvt_panel.update(
        #             input_water_temperature,
        #             current_weather,
        #         )  # [K]
        #     else:
        #         pvt_update_output = tuple(
        #             sum(entry)
        #             for entry in zip(
        #                 pvt_update_output,
        #                 pvt_panel.update(
        #                     input_water_temperature,
        #                     current_weather,
        #                 ),
        #             )
        #         )
        # <<< Code in this next block runs the PVT updator at the internal resolution.
        pvt_update_output = pvt_panel.update(
            input_water_temperature, parsed_args.internal_resolution, current_weather
        )

        (
            back_plate_heat_loss,  # [W]
            bulk_water_heat_gain,  # [W]
            collector_heat_input,  # [W]
            output_water_temperature,  # [K]
            upward_collector_heat_loss,  # [W]
            upward_glass_heat_loss,  # [W]
        ) = pvt_update_output

        # Turn off the hot-water pump if the HTF is losing heat to the panel.
        # if bulk_water_heat_gain <= 0:
        #     htf_pump.on = False
        # else:
        #     htf_pump.on = True

        # Propogate this information through to the heat exchanger and pass in the
        # tank s.t. it updates the tank correctly as well.
        # The tank heat gain here is measured in Joules.
        (
            updated_input_water_temperature,  # [K]
            tank_heat_gain,  # [J]
        ) = heat_exchanger.update(
            input_water_heat_capacity=pvt_panel.htf_heat_capacity,  # [J/kg*K]
            input_water_mass=pvt_panel.mass_flow_rate  # [kg/s]
            * parsed_args.internal_resolution,  # [s]
            input_water_temperature=output_water_temperature,  # [K]
            water_tank=hot_water_tank,
        )

        # Compute the new tank temperature after supplying this demand
        tank_output_water_temp = hot_water_tank.update(  # [K]
            tank_heat_gain,  # [J]
            parsed_args.internal_resolution,  # [minutes]
            current_hot_water_load,  # [litres/time_step] = [kg/time_step]
            weather_forecaster.mains_water_temp,  # [K]
            INTERNAL_HOUSEHOLD_AMBIENT_TEMPERATURE,  # [K]
        )

        # Determine various efficiency factors
        # The auxiliary heating will be measured in Watts.
        auxiliary_heating = (
            current_hot_water_load  # [litres/time_step] = [kg/time_step]
            * HEAT_CAPACITY_OF_WATER  # [J/kg*K]
            * (HOT_WATER_DEMAND_TEMP - tank_output_water_temp)  # [K]
        ) / (  # [J]
            parsed_args.internal_resolution  # [s]
        )  # [W]

        dc_electrical = (
            efficiency.dc_electrical(
                electrical_output=pvt_panel.electrical_output(current_weather),
                electrical_losses=htf_pump.power if htf_pump.on else 0,
                electrical_demand=current_electrical_load,
            )
            * 100
        )  # [%]

        thermal_output = (
            current_hot_water_load
            * HEAT_CAPACITY_OF_WATER
            * (tank_output_water_temp - weather_forecaster.mains_water_temp)
        )
        thermal_demand = (
            current_hot_water_load
            * HEAT_CAPACITY_OF_WATER
            * (HOT_WATER_DEMAND_TEMP - weather_forecaster.mains_water_temp)
        )

        dc_thermal = (
            efficiency.dc_thermal(
                thermal_output=thermal_output, thermal_demand=thermal_demand
            )
            * 100
        )  # [%]

        # Store the information in the dictionary mapping between time step and data.
        system_data_entry = {
            run_number: SystemData(
                ambient_temperature=current_weather.ambient_temperature
                - ZERO_CELCIUS_OFFSET,
                auxiliary_heating=auxiliary_heating,
                back_plate_heat_loss=back_plate_heat_loss
                / parsed_args.internal_resolution,
                bulk_water_heat_gain=bulk_water_heat_gain
                / parsed_args.internal_resolution,
                bulk_water_temperature=pvt_panel.bulk_water_temperature
                - ZERO_CELCIUS_OFFSET,
                collector_heat_input=collector_heat_input
                / parsed_args.internal_resolution,
                collector_input_temperature=input_water_temperature
                - ZERO_CELCIUS_OFFSET,
                collector_output_temperature=pvt_panel.collector_output_temperature
                - ZERO_CELCIUS_OFFSET,
                collector_temperature=pvt_panel.collector_temperature
                - ZERO_CELCIUS_OFFSET,
                collector_temperature_gain=pvt_panel.collector_output_temperature
                - input_water_temperature,
                date=datetime.date.strftime(date_and_time, "%d/%m/%y"),
                dc_electrical=dc_electrical,
                dc_thermal=dc_thermal,
                electrical_load=current_electrical_load,
                exchanger_temperature_drop=pvt_panel.collector_output_temperature
                - updated_input_water_temperature,
                glass_temperature=pvt_panel.glass_temperature - ZERO_CELCIUS_OFFSET,  # type: ignore
                gross_electrical_output=pvt_panel.electrical_output(current_weather),
                htf_pump_on=htf_pump.on,
                hot_water_load=load_system[(load.ProfileType.HOT_WATER, date_and_time)],
                net_electrical_output=pvt_panel.electrical_output(current_weather)
                - current_electrical_load,
                normal_irradiance=pvt_panel.get_solar_irradiance(current_weather),
                pv_efficiency=pvt_panel.electrical_efficiency,
                pv_temperature=pvt_panel.pv_temperature - ZERO_CELCIUS_OFFSET  # type: ignore
                if not parsed_args.no_pv
                else None,
                sky_temperature=current_weather.sky_temperature - ZERO_CELCIUS_OFFSET,
                solar_irradiance=current_weather.irradiance,
                tank_temperature=hot_water_tank.temperature - ZERO_CELCIUS_OFFSET,
                tank_heat_addition=tank_heat_gain / (parsed_args.internal_resolution),
                tank_output_temperature=tank_output_water_temp - ZERO_CELCIUS_OFFSET,
                thermal_load=current_hot_water_load
                * HEAT_CAPACITY_OF_WATER
                * 50
                / (parsed_args.internal_resolution),
                thermal_output=current_hot_water_load
                * HEAT_CAPACITY_OF_WATER
                * (tank_output_water_temp - weather_forecaster.mains_water_temp)
                / (parsed_args.internal_resolution),
                time=datetime.date.strftime(date_and_time, "%H:%M:%S"),
                upward_collector_heat_loss=upward_collector_heat_loss
                / parsed_args.internal_resolution,
                upward_glass_heat_loss=upward_glass_heat_loss
                / parsed_args.internal_resolution,
            )
        }

        total_power_data.increment(
            pvt_panel.electrical_output(current_weather)
            * parsed_args.internal_resolution,
            current_electrical_load * parsed_args.internal_resolution,
            thermal_output,
            thermal_demand,
        )

        # Dump the data generated to the output YAML file.
        _save_data(FileType.YAML, parsed_args.output, system_data_entry)
        system_data[run_number] = system_data_entry[run_number]

        # Cycle around the water if the pump is on, otherwise there is no gain.
        if htf_pump.on:
            input_water_temperature = updated_input_water_temperature

        # If at the end of an hour, dump the data.
        if parsed_args.internal_resolution <= 60:
            if date_and_time.minute == (
                60 - math.ceil(parsed_args.internal_resolution / 60)
            ) and date_and_time.second == (60 - parsed_args.internal_resolution):
                # Compute the carbon emissions and savings.
                carbon_emissions = mains_supply.get_carbon_emissions(total_power_data)
                # Append the data dump to the file.
                _save_data(
                    FileType.JSON,
                    parsed_args.output,
                    system_data,
                    carbon_emissions,
                    total_power_data,
                )
                # Clear the current system data store.
                system_data = dict()
        else:
            # Compute the carbon emissions and savings.
            carbon_emissions = mains_supply.get_carbon_emissions(total_power_data)
            # Append the data dump to the file.
            _save_data(
                FileType.JSON,
                parsed_args.output,
                system_data,
                carbon_emissions,
                total_power_data,
            )
            # Clear the current system data store.
            system_data = dict()


if __name__ == "__main__":
    main(sys.argv[1:])
