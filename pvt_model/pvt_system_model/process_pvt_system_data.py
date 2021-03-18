#!/usr/bin/python3.7
########################################################################################
# process_pvt_system_data.py - Module for processing the PVT data files.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The module for processing the PVT data files.

This module exposes various methods for instantiating and returning instances of the
classess representing physical components of the PVT system based on paths to various
data files.

Processing of these files, and extraction of their data to instantiate the components,
happens within this module.

"""

import datetime

from math import ceil
from typing import Any, Dict, Optional, Tuple

from .pvt_panel import adhesive, bond, eva, pvt, segment, tedlar
from . import exchanger, tank, pump

from ..__utils__ import MissingParametersError
from .constants import (
    HEAT_CAPACITY_OF_WATER,
)
from .__utils__ import (
    CollectorParameters,
    InvalidDataError,
    MissingDataError,
    OpticalLayerParameters,
    PVParameters,
    read_yaml,
)
from .index_handler import x_coordinate, y_coordinate
from .pvt_panel.__utils__ import MicroLayer


__all__ = (
    "heat_exchanger_from_path",
    "hot_water_tank_from_path",
    "pump_from_path",
    "pvt_panel_from_path",
)

#######################
# Heat exchanger code #
#######################


def heat_exchanger_from_path(exchanger_data_file: str) -> exchanger.Exchanger:
    """
    Generate a :class:`exchanger.Exchanger` instance based on the path to the data file.

    :param exchanger_data_file:
        The path to the exchanger data file.

    :return:
        A :class:`exchanger.Exchanger` instance representing the heat exchanger.

    """

    exchanger_data = read_yaml(exchanger_data_file)
    try:
        return exchanger.Exchanger(float(exchanger_data["efficiency"]))  # [unitless]
    except KeyError as e:
        raise MissingDataError(
            f"The file '{exchanger_data_file}' "
            f"is missing exchanger parameters: efficiency: {str(e)}"
        ) from None
    except ValueError as e:
        raise InvalidDataError(
            exchanger_data_file,
            "The exchanger efficiency must be a floating point integer.",
        ) from None


#######################
# Hot water tank code #
#######################


def hot_water_tank_from_path(tank_data_file: str) -> tank.Tank:
    """
    Generate a :class:`tank.Tank` instance based on the path to the data file.

    :param tank_data_file:
        The path to the tank data file.

    :param mains_water_temp:
        The mains-water temperature, measured in Kelvin.

    :return:
        A :class:`tank.Tank` instance representing the hot-water tank.

    """

    tank_data = read_yaml(tank_data_file)
    try:
        return tank.Tank(
            float(tank_data["area"]),  # [m^2]
            float(tank_data["diameter"]),  # [m]
            HEAT_CAPACITY_OF_WATER,  # [J/kg*K]
            float(tank_data["heat_loss_coefficient"]),  # [W/m^2*K]
            float(tank_data["mass"]),  # [kg]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all data needed to instantiate the tank class was provided. "
            f"File: {tank_data_file}. Error: {str(e)}"
        ) from None
    except ValueError as e:
        raise InvalidDataError(
            tank_data_file,
            "Tank data variables provided must be floating point integers.",
        ) from None


#############
# Pump code #
#############


def pump_from_path(pump_data_file: str) -> pump.Pump:
    """
    Generate a :class:`pump.Pump` instance based on the path to the data file.

    :param pump)data_file:
        The path to the pump data file.

    :return:
        A :class:`tank.Tank` instance representing the hot-water tank.

    """

    pump_data = read_yaml(pump_data_file)
    try:
        return pump.Pump(
            float(pump_data["power"]),  # [W]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all data needed to instantiate the pump class was provided. "
            f"File: {pump_data_file}. Error: {str(e)}"
        ) from None
    except ValueError as e:
        raise InvalidDataError(
            pump_data_file,
            "Pump data variables provided must be floating point integers.",
        ) from None


##################
# PVT panel code #
##################


def _collector_params_from_data(
    length: float,
    collector_data: Dict[str, Any],
) -> CollectorParameters:
    """
    Generate a :class:`CollectorParameters` containing collector-layer info from data.

    The HTF is assumed to be water unless the HTF heat capacity is supplied in the
    collector data.

    :param area:
        The area of the PV-T system, measured in meters squared.

    :param length:
        The length of the PV-T system, measured in meters.

    :param initial_collector_htf_tempertaure:
        The initial temperature of heat-transfer fluid in the collector.

    :param collector_data:
        The raw collector data extracted from the YAML data file.

    :return:
        The collector data, as a :class:`__utils__.CollectorParameters`, ready to
        instantiate a :class:`pvt.Collector` layer instance.

    """

    try:
        return CollectorParameters(
            absorptivity=collector_data["absorptivity"],  # [unitless]
            conductivity=collector_data["thermal_conductivity"]
            if "thermal_conductivity" in collector_data
            else None,
            density=collector_data["density"] if "density" in collector_data else None,
            emissivity=collector_data["emissivity"],  # [unitless]
            heat_capacity=collector_data["heat_capacity"],  # [J/kg*K]
            htf_heat_capacity=collector_data["htf_heat_capacity"]  # [J/kg*K]
            if "htf_heat_capacity" in collector_data
            else HEAT_CAPACITY_OF_WATER,  # [J/kg*K]
            inner_pipe_diameter=collector_data["inner_pipe_diameter"],  # [m]
            length=length,  # [m]
            mass_flow_rate=collector_data["mass_flow_rate"],  # [Litres/hour]
            number_of_pipes=collector_data["number_of_pipes"],  # [pipes]
            outer_pipe_diameter=collector_data["outer_pipe_diameter"],  # [m]
            thickness=collector_data["thickness"],  # [m]
            transmissivity=collector_data["transmissivity"],  # [unitless]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed collector-layer data provided. Potential problem: collector"
            "mass must be specified, either as 'mass' or 'density' and 'area' and "
            f"'thickness' params: {str(e)}"
        ) from None


def _glass_params_from_data(
    glass_data: Dict[str, Any]
) -> Tuple[float, OpticalLayerParameters]:
    """
    Generate a :class:`OpticalLayerParameters` containing glass-layer info from data.

    :param area:
        The area of the PV-T system, measured in meters squared.

    :param glass_data:
        The raw glass data extracted from the YAML data file.

    :return:
        The glass data, as a :class:`__utils__.OpticalLayerParameters`, ready to
        instantiate a :class:`pvt.Glass` layer instance.

    """

    try:
        return (
            glass_data["diffuse_reflection_coefficient"],
            OpticalLayerParameters(
                absorptivity=glass_data["absorptivity"],  # [unitless]
                conductivity=glass_data["thermal_conductivity"]
                if "thermal_conductivity" in glass_data
                else None,
                density=glass_data["density"],  # [kg/m^3]
                emissivity=glass_data["emissivity"],  # [unitless]
                heat_capacity=glass_data["heat_capacity"],  # [J/kg*K]
                thickness=glass_data["thickness"],  # [m]
                transmissivity=glass_data["transmissivity"],  # [unitless]
            ),
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed glass-layer data provided. Potential problem: Glass mass "
            "must be specified, either as 'mass' or 'density' and 'area' and "
            f"'thickness' params: {str(e)}"
        ) from None


def _pv_params_from_data(pv_data: Optional[Dict[str, Any]]) -> PVParameters:
    """
    Generate a :class:`PVParameters` containing PV-layer info from data.

    :param area:
        The area of the PV-T system, measured in meters squared.

    :param pv_data:
        The raw PV data extracted from the YAML data file.

    :return:
        The PV data, as a :class:`__utils__.PVParameters`, ready to instantiate a
        :class:`pvt.PV` layer instance.

    """

    if pv_data is None:
        raise MissingParametersError(
            "PV", "PV parameters must be specified in the PVT YAML file."
        )

    try:
        return PVParameters(
            conductivity=pv_data["thermal_conductivity"]
            if "thermal_conductivity" in pv_data
            else None,
            density=pv_data["density"] if "density" in pv_data else None,  # [kg/m^3]
            heat_capacity=pv_data["heat_capacity"],  # [J/kg*K]
            thickness=pv_data["thickness"],  # [m]
            transmissivity=pv_data["transmissivity"],  # [unitless]
            absorptivity=pv_data["absorptivity"],  # [unitless]
            emissivity=pv_data["emissivity"],  # [unitless]
            reference_efficiency=pv_data["reference_efficiency"],  # [unitless]
            reference_temperature=pv_data["reference_temperature"],  # [K]
            thermal_coefficient=pv_data["thermal_coefficient"],  # [K^-1]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed PV-layer data provided. Potential problem: PV mass must be"
            "specified, either as 'mass' or 'density' and 'area' and 'thickness' "
            f"params: {str(e)}"
        ) from None


def pvt_panel_from_path(
    portion_covered: float,
    pvt_data_file: str,
    x_resolution: int,
    y_resolution: int,
) -> pvt.PVT:
    """
    Generate a :class:`pvt.PVT` instance based on the path to the data file.

    :param portion_covered:
        The portion of the PV-T panel which is covered in PV.

    :param pvt_data_file:
        The path to the pvt data file.

    :param x_resolution:
        The x resolution of the simulation being run.

    :param y_resolution:
        The y resolution of the simulation being run.

    :return:
        A :class:`pvt.PVT` instance representing the PVT panel.

    """

    # Parse the data file into the various data classes.
    pvt_data = read_yaml(pvt_data_file)
    diffuse_reflection_coefficient, glass_parameters = _glass_params_from_data(
        pvt_data["glass"]
    )
    pv_parameters = _pv_params_from_data(pvt_data["pv"] if "pv" in pvt_data else None)
    collector_parameters = _collector_params_from_data(
        pvt_data["pvt_system"]["length"],  # [m]
        pvt_data["collector"],
    )

    # Construct the segmented array based on the arguments.
    pipe_positions = list(
        range(
            x_resolution // (pvt_data["collector"]["number_of_pipes"] + 1),
            x_resolution,
            ceil(x_resolution / (pvt_data["collector"]["number_of_pipes"] + 1)),
        )
    )
    pv_coordinate_cutoff = int(y_resolution * portion_covered)
    try:
        segments = {
            segment.SegmentCoordinates(
                x_coordinate(segment_number, x_resolution),
                y_coordinate(segment_number, x_resolution),
            ): segment.Segment(
                True,
                True,
                pvt_data["pvt_system"]["length"] / x_resolution,
                x_coordinate(segment_number, x_resolution) in pipe_positions,
                y_coordinate(segment_number, x_resolution) <= pv_coordinate_cutoff,
                pvt_data["pvt_system"]["width"] / y_resolution,
                x_coordinate(segment_number, x_resolution),
                y_coordinate(segment_number, x_resolution),
                pipe_positions.index(x_coordinate(segment_number, x_resolution))
                if x_coordinate(segment_number, x_resolution) in pipe_positions
                else None,
            )
            for segment_number in range(x_resolution * y_resolution)
        }
    except KeyError as e:
        raise MissingParametersError(
            "PVT", f"Missing parameters when instantiating the PV-T system: {str(e)}"
        ) from None

    try:
        pvt_panel = pvt.PVT(
            absorber_pipe_bond=bond.Bond(
                pvt_data["bond"]["thermal_conductivity"],
                pvt_data["bond"]["thickness"],
                pvt_data["bond"]["width"],
            ),
            adhesive=adhesive.Adhesive(
                pvt_data["adhesive"]["thermal_conductivity"],
                pvt_data["adhesive"]["thickness"],
            ),
            air_gap_thickness=pvt_data["air_gap"]["thickness"],  # [m]
            area=pvt_data["pvt_system"]["area"]
            if "area" in pvt_data["pvt_system"]
            else pvt_data["pvt_system"]["width"]
            * pvt_data["pvt_system"]["length"],  # [m^2]
            collector_parameters=collector_parameters,
            diffuse_reflection_coefficient=diffuse_reflection_coefficient,
            eva=eva.EVA(
                pvt_data["eva"]["thermal_conductivity"], pvt_data["eva"]["thickness"]
            ),
            glass_parameters=glass_parameters,
            insulation=MicroLayer(
                pvt_data["insulation"]["thermal_conductivity"],
                pvt_data["insulation"]["thickness"],
            ),
            latitude=pvt_data["pvt_system"]["latitude"],  # [deg]
            length=pvt_data["pvt_system"]["length"],  # [m]
            longitude=pvt_data["pvt_system"]["longitude"],  # [deg]
            portion_covered=portion_covered,  # [unitless]
            pv_parameters=pv_parameters,
            pv_to_collector_thermal_conductance=pvt_data["pvt_system"][
                "pv_to_collector_conductance"
            ],  # [W/m^2*K]
            segments=segments,
            tedlar=tedlar.Tedlar(
                pvt_data["tedlar"]["thermal_conductivity"],
                pvt_data["tedlar"]["thickness"],
            ),
            timezone=datetime.timezone(
                datetime.timedelta(hours=int(pvt_data["pvt_system"]["timezone"]))
            ),
            width=pvt_data["pvt_system"]["width"],  # [m]
            azimuthal_orientation=pvt_data["pvt_system"][
                "azimuthal_orientation"
            ]  # [deg]
            if "azimuthal_orientation" in pvt_data["pvt_system"]
            else None,
            horizontal_tracking=pvt_data["pvt_system"]["horizontal_tracking"],
            vertical_tracking=pvt_data["pvt_system"]["vertical_tracking"],
            tilt=pvt_data["pvt_system"]["tilt"]  # [deg]
            if "tilt" in pvt_data["pvt_system"]
            else None,
        )
    except KeyError as e:
        raise MissingParametersError(
            "PVT", f"Missing parameters when instantiating the PV-T system: {str(e)}"
        ) from None
    except TypeError as e:
        raise InvalidDataError(
            "PVT Data File", f"Error parsing data types - type mismatch: {str(e)}"
        ) from None

    return pvt_panel
