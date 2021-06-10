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

from logging import Logger
from typing import Any, Dict, Optional, Set

from .pvt_collector import adhesive, bond, eva, pvt, element, tedlar
from . import exchanger, tank, pump

from ..__utils__ import (
    InvalidParametersError,
    MissingParametersError,
    read_yaml,
    TemperatureName,
)
from .constants import (
    EDGE_LENGTH,
    EDGE_WIDTH,
    HEAT_CAPACITY_OF_WATER,
)
from .__utils__ import (
    CollectorParameters,
    InvalidDataError,
    MissingDataError,
    OpticalLayerParameters,
    PVParameters,
)
from .index_handler import x_coordinate, y_coordinate
from .pvt_collector.__utils__ import MicroLayer


__all__ = (
    "heat_exchanger_from_path",
    "hot_water_tank_from_path",
    "pump_from_path",
    "pvt_collector_from_path",
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


def _absorber_params_from_data(
    length: float,
    absorber_data: Dict[str, Any],
    override_mass_flow_rate: Optional[float],
) -> CollectorParameters:
    """
    Generate a :class:`CollectorParameters` containing absorber-layer info from data.

    The HTF is assumed to be water unless the HTF heat capacity is supplied in the
    absorber data.

    :param length:
        The length of the PV-T system, measured in meters.

    :param absorber_data:
        The raw absorber data extracted from the YAML data file.

    :param override_mass_flow_rate:
        If specified, this overrides the mass flow-rate in the absorber. The units are
        litres per hour.

    :return:
        The absorber data, as a :class:`__utils__.CollectorParameters`, ready to
        instantiate a :class:`pvt.Collector` layer instance.

    """

    # import pdb

    # pdb.set_trace()

    try:
        return CollectorParameters(
            absorptivity=absorber_data["absorptivity"],  # [unitless]
            conductivity=absorber_data["thermal_conductivity"]
            if "thermal_conductivity" in absorber_data
            else None,
            density=absorber_data["density"] if "density" in absorber_data else None,
            emissivity=absorber_data["emissivity"],  # [unitless]
            heat_capacity=absorber_data["heat_capacity"],  # [J/kg*K]
            htf_heat_capacity=absorber_data["htf_heat_capacity"]  # [J/kg*K]
            if "htf_heat_capacity" in absorber_data
            else HEAT_CAPACITY_OF_WATER,  # [J/kg*K]
            inner_pipe_diameter=absorber_data["inner_pipe_diameter"],  # [m]
            length=length,  # [m]
            mass_flow_rate=override_mass_flow_rate
            if override_mass_flow_rate is not None
            else absorber_data["mass_flow_rate"],  # [Litres/hour]
            number_of_pipes=absorber_data["number_of_pipes"],  # [pipes]
            outer_pipe_diameter=absorber_data["outer_pipe_diameter"],  # [m]
            pipe_density=absorber_data["pipe_density"],  # [kg/m^3]
            thickness=absorber_data["thickness"],  # [m]
            transmissivity=absorber_data["transmissivity"],  # [unitless]
        )
    except KeyError as e:
        raise MissingDataError(
            "Not all needed absorber-layer data provided. Potential problem: absorber"
            "mass must be specified, either as 'mass' or 'density' and 'area' and "
            f"'thickness' params: {str(e)}"
        ) from None


def _glass_params_from_data(glass_data: Dict[str, Any]) -> OpticalLayerParameters:
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
        return OpticalLayerParameters(
            absorptivity=glass_data["absorptivity"],  # [unitless]
            conductivity=glass_data["thermal_conductivity"]
            if "thermal_conductivity" in glass_data
            else None,
            density=glass_data["density"],  # [kg/m^3]
            emissivity=glass_data["emissivity"],  # [unitless]
            heat_capacity=glass_data["heat_capacity"],  # [J/kg*K]
            thickness=glass_data["thickness"],  # [m]
            transmissivity=glass_data["transmissivity"],  # [unitless]
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


def _elements_from_data(
    edge_length: float,
    edge_width: float,
    layers: Set[TemperatureName],
    logger: Logger,
    portion_covered: float,
    pvt_data: Dict[Any, Any],
    x_resolution: int,
    y_resolution: int,
) -> Any:
    """
    Returns mapping from element coordinate to element based on the input data.

    :param edge_length:
        The maximum length of an edge element along the top and bottom edges of the
        panel, measured in meters.

    :param edge_width:
        The maximum width of an edge element along the side edges of the panel, measured
        in meters.

    :param layers:
        The `set` of layers to include in the system.

    :param logger:
        The :class:`logging.Logger` logger instance used for the run.

    :param portion_covered:
        The portion of the PVT absorber that is covered with PV cells. The uncovered
        section is mapped as solar absorber only with glazing as appropriate.

    :param pvt_data:
        The raw PVT data, extracted from the data file.

    :param x_resolution:
        The x resolution for the run.

    :param y_resolution:
        The y resolution for the run.

    :return:
        A mapping between the element coordinates and the element for all elements
        within the panel.

    """

    # * If 1x1, warn that 1x1 resolution is depreciated and should not really be used.
    if x_resolution == 1 and y_resolution == 1:
        logger.warn(
            "Running the system at a 1x1 resolution is depreciated. Consider running "
            "at a higher resolution."
        )
        return {
            element.ElementCoordinates(0, 0): element.Element(
                TemperatureName.absorber in layers,
                TemperatureName.glass in layers,
                pvt_data["pvt_collector"]["length"],
                True,
                TemperatureName.pv in layers,
                TemperatureName.upper_glass in layers,
                pvt_data["pvt_collector"]["width"],
                0,
                0,
                0,
            )
        }

    # Extract the necessary parameters from the system data.
    try:
        number_of_pipes = pvt_data["absorber"]["number_of_pipes"]
    except KeyError as e:
        raise MissingParametersError(
            "Element", "The number of pipes attached to the absorber must be supplied."
        ) from None
    try:
        panel_length = pvt_data["pvt_collector"]["length"]
    except KeyError as e:
        raise MissingParametersError(
            "Element", "PVT panel length must be supplied."
        ) from None

    try:
        panel_width = pvt_data["pvt_collector"]["width"]
    except KeyError as e:
        raise MissingParametersError(
            "Element", "PVT panel width must be supplied."
        ) from None
    try:
        bond_width = pvt_data["bond"]["width"]
    except KeyError as e:
        raise MissingParametersError(
            "Element", "Collector-to-pipe bond width must be supplied."
        ) from None

    # * Determine the spacing between the pipes.
    pipe_spacing = (x_resolution - number_of_pipes) / (number_of_pipes + 1)
    if int(pipe_spacing) != pipe_spacing:
        raise InvalidParametersError(
            "The resolution supplied results in an uneven pipe distribution.",
            "pipe_spcaing",
        )

    # * Determine the indicies of elements that have pipes attached.
    pipe_positions = list(
        range(int(pipe_spacing), x_resolution - 2, int(pipe_spacing) + 1)
    )

    # Determine whether the width of the elements is greater than or less than the edge
    # width and adjust accordingly.
    nominal_element_width: float = (
        panel_width - number_of_pipes * bond_width - 2 * edge_width
    ) / (x_resolution - number_of_pipes - 2)
    if nominal_element_width < edge_width:
        nominal_element_width = (panel_width - number_of_pipes * bond_width) / (
            x_resolution - number_of_pipes
        )
        edge_width = nominal_element_width

    # Likewise, determine whether the nominal element height is greater than the edge
    # height and adjust accordingly.
    nominal_element_length: float = (panel_length - 2 * edge_length) / (
        y_resolution - 2
    )
    if nominal_element_length < edge_length:
        nominal_element_length = panel_length / y_resolution
        edge_length = nominal_element_length

    # * Instantiate the array of elements.

    # Construct the elemented array based on the arguments.
    pv_coordinate_cutoff = int(y_resolution * portion_covered)
    try:
        elements = {
            element.ElementCoordinates(
                x_coordinate(element_number, x_resolution),
                y_coordinate(element_number, x_resolution),
            ): element.Element(
                absorber=TemperatureName.absorber in layers,
                glass=TemperatureName.glass in layers,
                length=edge_length
                if y_coordinate(element_number, x_resolution) in {0, y_resolution - 1}
                else nominal_element_length,
                pipe=x_coordinate(element_number, x_resolution) in pipe_positions
                if TemperatureName.pipe in layers
                else False,
                pv=y_coordinate(element_number, x_resolution) <= pv_coordinate_cutoff
                if TemperatureName.pv in layers
                else False,
                upper_glass=TemperatureName.upper_glass in layers,
                # Use the edge with if the element is an edge element.
                width=edge_width
                if x_coordinate(element_number, x_resolution) in {0, x_resolution - 1}
                # Otherwise, use the bond width if the element is a pipe element.
                else bond_width
                if x_coordinate(element_number, x_resolution) in pipe_positions
                # Otherwise, use the nominal element width.
                else nominal_element_width,
                x_index=x_coordinate(element_number, x_resolution),
                y_index=y_coordinate(element_number, x_resolution),
                pipe_index=pipe_positions.index(
                    x_coordinate(element_number, x_resolution)
                )
                if x_coordinate(element_number, x_resolution) in pipe_positions
                else None,
            )
            for element_number in range(x_resolution * y_resolution)
        }
    except KeyError as e:
        raise MissingParametersError(
            "PVT", f"Missing parameters when instantiating the PV-T system: {str(e)}"
        ) from None

    return elements


def pvt_collector_from_path(
    layers: Set[TemperatureName],
    logger: Logger,
    override_mass_flow_rate: Optional[float],
    portion_covered: float,
    pvt_data_file: str,
    x_resolution: int,
    y_resolution: int,
) -> pvt.PVT:
    """
    Generate a :class:`pvt.PVT` instance based on the path to the data file.

    :param layers:
        The `set` of layers that should be included in the panel.

    :param logger:
        The :class:`logging.Logger` used for the run.

    :param override_mass_flow_rate:
        If specified, the mass-flow rate in the collector uses this value rather than
        the value in the data file. The units are litres per hour.

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

    # If there is upper-glass data present, process this accordingly.
    if "upper_glass" in pvt_data and TemperatureName.upper_glass in layers:
        upper_glass_parameters: Optional[
            OpticalLayerParameters
        ] = _glass_params_from_data(pvt_data["upper_glass"])
    elif "glass" in pvt_data and TemperatureName.upper_glass in layers:
        upper_glass_parameters = _glass_params_from_data(pvt_data["glass"])
    elif TemperatureName.upper_glass in layers:
        raise InvalidParametersError(
            "Upper glass layer requested on the command line but no glass data in pvt "
            "data file.",
            "glass",
        )
    else:
        upper_glass_parameters = None

    # If there is glass data present, process this accordingly.
    if "glass" in pvt_data and TemperatureName.glass in layers:
        air_gap_thickness = pvt_data["air_gap"]["thickness"]  # [m]
        glass_parameters: Optional[OpticalLayerParameters] = _glass_params_from_data(
            pvt_data["glass"]
        )
    elif TemperatureName.glass in layers:
        raise InvalidParametersError(
            "Glass layer requested on the command line but no glass data in pvt data "
            "file.",
            "glass",
        )
    else:
        air_gap_thickness = None
        glass_parameters = None

    pv_parameters = _pv_params_from_data(pvt_data["pv"] if "pv" in pvt_data else None)
    absorber_parameters = _absorber_params_from_data(
        pvt_data["pvt_collector"]["length"],  # [m]
        pvt_data["absorber"],
        override_mass_flow_rate,
    )

    elements = _elements_from_data(
        EDGE_LENGTH,
        EDGE_WIDTH,
        layers,
        logger,
        portion_covered,
        pvt_data,
        x_resolution,
        y_resolution,
    )

    try:
        pvt_collector = pvt.PVT(
            absorber_pipe_bond=bond.Bond(
                pvt_data["bond"]["thermal_conductivity"],
                pvt_data["bond"]["thickness"],
                pvt_data["bond"]["width"],
            ),
            adhesive=adhesive.Adhesive(
                pvt_data["adhesive"]["thermal_conductivity"],
                pvt_data["adhesive"]["thickness"],
            ),
            air_gap_thickness=air_gap_thickness,
            area=pvt_data["pvt_collector"]["area"]
            if "area" in pvt_data["pvt_collector"]
            else pvt_data["pvt_collector"]["width"]
            * pvt_data["pvt_collector"]["length"],  # [m^2]
            absorber_parameters=absorber_parameters,
            eva=eva.EVA(
                pvt_data["eva"]["thermal_conductivity"], pvt_data["eva"]["thickness"]
            ),
            glass_parameters=glass_parameters,
            insulation=MicroLayer(
                pvt_data["insulation"]["thermal_conductivity"],
                pvt_data["insulation"]["thickness"],
            ),
            latitude=pvt_data["pvt_collector"]["latitude"],  # [deg]
            length=pvt_data["pvt_collector"]["length"],  # [m]
            longitude=pvt_data["pvt_collector"]["longitude"],  # [deg]
            portion_covered=portion_covered,  # [unitless]
            pv_parameters=pv_parameters,
            elements=elements,
            tedlar=tedlar.Tedlar(
                pvt_data["tedlar"]["thermal_conductivity"],
                pvt_data["tedlar"]["thickness"],
            ),
            timezone=datetime.timezone(
                datetime.timedelta(hours=int(pvt_data["pvt_collector"]["timezone"]))
            ),
            upper_glass_parameters=upper_glass_parameters,
            width=pvt_data["pvt_collector"]["width"],  # [m]
            azimuthal_orientation=pvt_data["pvt_collector"][
                "azimuthal_orientation"
            ]  # [deg]
            if "azimuthal_orientation" in pvt_data["pvt_collector"]
            else None,
            horizontal_tracking=pvt_data["pvt_collector"]["horizontal_tracking"],
            vertical_tracking=pvt_data["pvt_collector"]["vertical_tracking"],
            tilt=pvt_data["pvt_collector"]["tilt"]  # [deg]
            if "tilt" in pvt_data["pvt_collector"]
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

    return pvt_collector
