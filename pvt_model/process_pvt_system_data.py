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

from typing import Any, Dict, Tuple

import numpy

from .pvt_panel import pvt
from . import exchanger, tank, pump
from .constants import (
    HEAT_CAPACITY_OF_WATER,
)
from .__utils__ import (
    BackLayerParameters,
    CollectorParameters,
    InvalidDataError,
    MissingDataError,
    MissingParametersError,
    OpticalLayerParameters,
    PVParameters,
    read_yaml,
)


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


def _back_params_from_data(
    area: float, back_data: Dict[str, Any]
) -> BackLayerParameters:
    """
    Generate a :class:`BackLayerParameters` containing back-layer info from data.

    :param area:
        The area of the PV-T system, measured in meters squared.

    :param back_data:
        The raw back data extracted from the YAML data file.

    :return:
        The back data, as a :class:`__utils__.BackLayerParameters`, ready to
        instantiate a :class:`pvt.BackPlater` layer instance.

    """

    try:
        return BackLayerParameters(
            back_data["mass"]  # [kg]
            if "mass" in back_data
            else back_data["density"]  # [kg/m^3]
            * area  # [m^2]
            * back_data["thickness"],  # [m]
            back_data["heat_capacity"],  # [J/kg*K]
            area,  # [m^2]
            back_data["thickness"],  # [m]
            back_data["thermal_conductivity"],  # [W/m*K]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed back-layer data provided. Potential problem: back-layer"
            "mass must be specified, either as 'mass' or 'density' and 'area' and "
            f"'thickness' params. Missing param: {str(e)}"
        ) from None


def _collector_params_from_data(
    area: float,
    length: float,
    initial_collector_htf_tempertaure: float,
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
            mass=collector_data["mass"]  # [kg]
            if "mass" in collector_data
            else (
                # Main collector body area.
                area  # [m^2]
                * collector_data["density"]  # [kg/m^3]
                * collector_data["thickness"]  # [m]
                # Collector pipe area
                + (
                    numpy.pi
                    * collector_data["pipe_diameter"]  # [m]
                    * collector_data["length"]  # [m]
                )
                * collector_data["density"]  # [kg/m^3]
                * collector_data["thickness"]  # [m]
            ),
            heat_capacity=collector_data["heat_capacity"],  # [J/kg*K]
            area=area,  # [m^2]
            thickness=collector_data["thickness"],  # [m]
            transmissivity=collector_data["transmissivity"],  # [unitless]
            absorptivity=collector_data["absorptivity"],  # [unitless]
            emissivity=collector_data["emissivity"],  # [unitless]
            bulk_water_temperature=initial_collector_htf_tempertaure,  # [K]
            htf_heat_capacity=collector_data["htf_heat_capacity"]  # [J/kg*K]
            if "htf_heat_capacity" in collector_data
            else HEAT_CAPACITY_OF_WATER,  # [J/kg*K]
            length=length,  # [m]
            mass_flow_rate=collector_data["mass_flow_rate"],  # [Litres/hour]
            number_of_pipes=collector_data["number_of_pipes"],  # [pipes]
            output_water_temperature=initial_collector_htf_tempertaure,  # [K]
            pipe_diameter=collector_data["pipe_diameter"],  # [m]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed collector-layer data provided. Potential problem: collector"
            "mass must be specified, either as 'mass' or 'density' and 'area' and "
            f"'thickness' params: {str(e)}"
        ) from None


def _glass_params_from_data(
    area: float, glass_data: Dict[str, Any]
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
                glass_data["mass"]  # [kg]
                if "mass" in glass_data
                else glass_data["density"]  # [kg/m^3]
                * glass_data["thickness"]  # [m]
                * area,  # [m^2]
                glass_data["heat_capacity"],  # [J/kg*K]
                area,  # [m^2]
                glass_data["thickness"],  # [m]
                glass_data["transmissivity"],  # [unitless]
                glass_data["absorptivity"],  # [unitless]
                glass_data["emissivity"],  # [unitless]
            ),
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed glass-layer data provided. Potential problem: Glass mass "
            "must be specified, either as 'mass' or 'density' and 'area' and "
            f"'thickness' params: {str(e)}"
        ) from None


def _pv_params_from_data(area: float, pv_data: Dict[str, Any]) -> PVParameters:
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

    try:
        return PVParameters(
            pv_data["mass"]  # [kg]
            if "mass" in pv_data
            else pv_data["density"]  # [kg/m^3]
            * area  # [m^2]
            * pv_data["thickness"],  # [m]
            pv_data["heat_capacity"],  # [J/kg*K]
            area,  # [m^2]
            pv_data["thickness"],  # [m]
            pv_data["transmissivity"],  # [unitless]
            pv_data["absorptivity"],  # [unitless]
            pv_data["emissivity"],  # [unitless]
            pv_data["reference_efficiency"],  # [unitless]
            pv_data["reference_temperature"],  # [K]
            pv_data["thermal_coefficient"],  # [K^-1]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed PV-layer data provided. Potential problem: PV mass must be"
            "specified, either as 'mass' or 'density' and 'area' and 'thickness' "
            f"params: {str(e)}"
        ) from None


def pvt_panel_from_path(
    initial_collector_htf_tempertaure: float,
    portion_covered: float,
    pvt_data_file: str,
    unglazed: bool,
) -> pvt.PVT:
    """
    Generate a :class:`pvt.PVT` instance based on the path to the data file.

    :param portion_covered:
        The portion of the PV-T panel which is covered in PV.

    :param pvt_data_file:
        The path to the pvt data file.

    :param ungalzed:
        Whether or not a glass layer (ie, glazing) is included in the panel. If set to
        `True`, then no glass layer is used.

    :return:
        A :class:`pvt.PVT` instance representing the PVT panel.

    """

    # Set up the PVT module
    pvt_data = read_yaml(pvt_data_file)

    diffuse_reflection_coefficient, glass_parameters = _glass_params_from_data(
        pvt_data["pvt_system"]["area"], pvt_data["glass"]
    )
    pv_parameters = (
        _pv_params_from_data(pvt_data["pvt_system"]["area"], pvt_data["pv"])
        if "pv" in pvt_data
        else None
    )
    collector_parameters = _collector_params_from_data(
        pvt_data["pvt_system"]["area"],  # [m^2]
        pvt_data["pvt_system"]["length"],  # [m]
        initial_collector_htf_tempertaure,  # [K]
        pvt_data["collector"],
    )
    back_parameters = _back_params_from_data(
        pvt_data["pvt_system"]["area"], pvt_data["back"]
    )

    try:
        pvt_panel = pvt.PVT(
            air_gap_thickness=pvt_data["air_gap"]["thickness"],  # [m]
            area=pvt_data["pvt_system"]["area"],  # [m^2]
            back_params=back_parameters,
            collector_parameters=collector_parameters,
            diffuse_reflection_coefficient=diffuse_reflection_coefficient,
            glass_parameters=glass_parameters,
            glazed=not unglazed,
            latitude=pvt_data["pvt_system"]["latitude"],  # [deg]
            longitude=pvt_data["pvt_system"]["longitude"],  # [deg]
            portion_covered=portion_covered,  # [unitless]
            pv_parameters=pv_parameters if portion_covered != 0 else None,
            pv_to_collector_thermal_conductance=pvt_data["pvt_system"][
                "pv_to_collector_conductance"
            ],  # [W/m^2*K]
            timezone=datetime.timezone(
                datetime.timedelta(hours=int(pvt_data["pvt_system"]["timezone"]))
            ),
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
