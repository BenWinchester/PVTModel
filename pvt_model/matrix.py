#!/usr/bin/python3.7
########################################################################################
# matrix.py - The matrix coefficient solver for the PV-T model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The matrix coefficient solver for the PV-T model.

The model works by arranging the system of differential equations as a matrix equation
such that the temperatures at each time step can be computed based on the coefficients
of the matrix which are computed based on the temperatures of the various components at
the previous time step, as well as various Physical and fundamental constants.

The equations in the system can be expressed as the matrix equation
    A * T = B
where
    A is the matrix computed and returned by this function;
    T is a vector containing the various temperatures;
    and B is the "resultant vector," computed elsewhere.

The equations represented by the rows of the matrix are:
    A[0]: The heat balance of the glass layer of the panel;
    A[1]: The heat balance of the PV layer of the panel;
    A[2]: The heat balance of the collector layer of the panel;
    A[3]: CoE via an NTU method of the HTF through the collector;
    A[4]: CoE via an NTU method of the HTF through the hot-water tank heat exchanger;
    A[5]: The heat balance of the hot-water tank.

The temperatures represented in the vector T are:
    T[0]: T_g     - the temperature of the glass layer,
    T[1]: T_pv    - the temperature of the PV layer,
    T[2]: T_c     - the temperature of the collector layer,
    T[3]: T_cin   - the temperature of HTF entering the collector,
    T[4]: T_cout  - the temperature of HTF leaving the collector,
    T[5]: T_t     - the temperature of the hot-water tank.
    (All values within the temperature vector are measured in Kelvin.);

"""

import numpy

from . import constants, physics_utils, tank

from .pvt_panel import pvt
from .__utils__ import WeatherConditions

__all__ = (
    "calculate_coefficient_matrix",
    "calculate_resultant_vector",
)


def _get_glass_equation_coefficients(
    previous_collector_temperature: float,
    previous_glass_temperature: float,
    previous_pv_temperature: float,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,
) -> numpy.ndarray:
    """
    Calculates the coefficient for the row representing the glass-layer equation.

    :param previous_collector_temperature:
        The temperature of the collector layer at the previous time step, measured in
        Kelvin.

    :param previous_glass_temperature:
        The temperature of the glass layer at the previous time step, measured in
        Kelvin.

    :param previous_pv_temperature:
        The temperature of the PV layer at the previous time step, measured in Kelvin.

    :param pvt_panel:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The resolution of the model being run, measured in seconds.

    :param weather_conditions:
        The weather conditions at the current time step.

    :return:
        An :class:`numpy.ndarray` containing the coefficients for the row in the matrix
        that represents the glass layer's equation. All values have units Watts per
        Kelvin.

    """

    # Instantiate an empty array to represent the matrix row.
    glass_equation_coefficients = numpy.zeros([1, 6])

    # Compute the glass temperature term.
    glass_equation_coefficients[0, 0] = (
        # Change in internal energy of the glass layer
        (
            pvt_panel.glass.mass  # [kg]
            * pvt_panel.glass.heat_capacity  # [J/kg*K]
            / resolution  # [s]
        )  # [W/s]
        # Conductive heat transfer with the PV and collector layers
        + physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        # Radiative heat transfer with the PV layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=previous_glass_temperature,
            source_emissivity=pvt_panel.pv.emissivity,
            source_temperature=previous_pv_temperature,
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Radiative heat transfer from the collector layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=previous_glass_temperature,
            source_emissivity=pvt_panel.collector.emissivity,
            source_temperature=previous_collector_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Conductive heat transfer to the wind
        + weather_conditions.wind_heat_transfer_coefficient  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        # Radiative heat transfer to the sky
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=previous_glass_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
    )  # [W/K]

    # Compute the PV temperature term.
    glass_equation_coefficients[0, 1] = -(
        # Conductive heat transfer from the PV layer
        physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Radiative heat transfer from the PV layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=previous_glass_temperature,
            source_emissivity=pvt_panel.pv.emissivity,
            source_temperature=previous_pv_temperature,
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
    )  # [W/K]

    # Compute the collector temperature term.
    glass_equation_coefficients[0, 2] = -(
        # Conductive heat transfer from the collector layer
        physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Radiative heat transfer from the collector layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=previous_glass_temperature,
            source_emissivity=pvt_panel.collector.emissivity,
            source_temperature=previous_collector_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
    )  # [W/K]

    return glass_equation_coefficients


def _get_pv_equation_coefficients(
    previous_glass_temperature: float,
    previous_pv_temperature: float,
    pvt_panel: pvt.PVT,
    resolution: int,
) -> numpy.ndarray:
    """
    Calculates the coefficient for the row representing the PV-layer equation.

    :param previous_glass_temperature:
        The temperature of the glass layer at the previous time step, measured in
        Kelvin.

    :param previous_pv_temperature:
        The temperature of the PV layer at the previous time step, measured in Kelvin.

    :param pvt_panel:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The resolution of the model being run, measured in seconds.

    :param weather_conditions:
        The weather conditions at the current time step.

    :return:
        An :class:`numpy.ndarray` containing the coefficients for the row in the matrix
        that represents the PV layer's equation. All values have units Watts per Kelvin.

    """

    pv_equation_coefficients = numpy.zeros([1, 6])

    # Compute the glass temperature term.
    pv_equation_coefficients[0, 0] = -(
        # Conductive heat transfer from the PV layer
        physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Radiative heat transfer from the PV layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.pv.emissivity,
            destination_temperature=previous_pv_temperature,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=previous_glass_temperature,
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
    )  # [W/K]

    # Compute the PV temperature term.
    pv_equation_coefficients[0, 1] = (
        # Change in internal energy of the PV layer.
        pvt_panel.pv.mass  # [kg]
        * pvt_panel.pv.heat_capacity  # [J/kg*K]
        / resolution  # [s]
        # Conductive heat transfer from the glass layer.
        + physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Radiative heat transfer from the glass layer.
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.pv.emissivity,
            destination_temperature=previous_pv_temperature,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=previous_glass_temperature,
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Conductive heat transfer to the collector layer.
        + pvt_panel.pv_to_collector_thermal_conductance  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
    )  # [W/K]

    # Compute the collector temperature term.
    pv_equation_coefficients[0, 2] = -(
        # Conductive heat transfer to the collector layer.
        pvt_panel.pv_to_collector_thermal_conductance  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
    )

    return pv_equation_coefficients


def _get_collector_equation_coefficients(
    collector_to_htf_efficiency: float,
    previous_collector_temperature: float,
    previous_glass_temperature: float,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,  # pylint: disable=unused-argument
) -> numpy.ndarray:
    """
    Calculates the coefficient for the row representing the collector-layer equation.

    :param collector_to_htf_efficiency:
        The efficiency of the heat transfer process between the thermal collector layer
        and the HTF in the collector tubes.

    :param previous_collector_temperature:
        The temperature of the collector layer at the previous time step, measured in
        Kelvin.

    :param previous_glass_temperature:
        The temperature of the glass layer at the previous time step, measured in
        Kelvin.

    :param pvt_panel:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The resolution of the model being run, measured in seconds.

    :param weather_conditions:
        The weather conditions at the current time step.

    :return:
        An :class:`numpy.ndarray` containing the coefficients for the row in the matrix
        that represents the collector layer's equation. All values have units Watts per
        Kelvin.

    """

    collector_equation_coefficients = numpy.zeros([1, 6])

    # Compute the glass temperature term.
    collector_equation_coefficients[0, 0] = -(
        # Conductive heat transfer from the glass layer
        physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Radiative heat transfer from the glass layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=previous_glass_temperature,
            source_emissivity=pvt_panel.collector.emissivity,
            source_temperature=previous_collector_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
    )

    # Compute the PV temperature term.
    collector_equation_coefficients[0, 1] = -(
        # Conductive heat transfer to the PV layer.
        pvt_panel.pv_to_collector_thermal_conductance  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
    )

    # Compute the collector temperature term.
    collector_equation_coefficients[0, 2] = (
        # Conductive heat transfer from the PV layer.
        pvt_panel.pv_to_collector_thermal_conductance  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Conductive heat transfer from the glass layer
        + physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Radiative heat transfer from the glass layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=previous_glass_temperature,
            source_emissivity=pvt_panel.collector.emissivity,
            source_temperature=previous_collector_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Heat transfer to the HTF.
        + collector_to_htf_efficiency
        * pvt_panel.collector.mass_flow_rate  # [kg/s]
        * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
        # Back plate heat loss.
        + pvt_panel.back_plate.conductance  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Internal energy change of the collector layer.
        + pvt_panel.collector.mass  # [kg]
        * pvt_panel.collector.heat_capacity  # [J/kg*K]
        * constants.NC
        / resolution  # [s]
    )  # [W/K]

    # Compute the collector-input-htf temperature term.
    collector_equation_coefficients[0, 3] = -(
        collector_to_htf_efficiency
        * pvt_panel.collector.mass_flow_rate  # [kg/s]
        * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
    )  # [W/K]

    return collector_equation_coefficients


def _get_collector_htf_equation_coefficients(
    collector_to_htf_efficiency: float,
) -> numpy.ndarray:
    """
    Calculates the coefficient for the row representing the collector's htf equation.

    :param collector_to_htf_efficiency:
        The efficiency of the heat transfer process between the thermal collector layer
        and the HTF in the collector tubes.

    :return:
        An :class:`numpy.ndarray` containing the coefficients for the row in the matrix
        that represents the collector layer's htf equation. All values are
        dimensionless.

    """

    collector_htf_equation_coefficients = numpy.zeros([1, 6])

    # Collector temperature term.
    collector_htf_equation_coefficients[0, 2] = collector_to_htf_efficiency
    # Collector input temperature term.
    collector_htf_equation_coefficients[0, 3] = 1 - collector_to_htf_efficiency
    # Collector output temperature term.
    collector_htf_equation_coefficients[0, 4] = -1

    return collector_htf_equation_coefficients


def _get_tank_htf_equation_coefficients(
    htf_to_tank_efficiency: float,
    previous_collector_output_temperature: float,
    previous_tank_temperature: float,
) -> numpy.ndarray:
    """
    Calculates the coefficient for the row representing the tank's htf equation.

    :param htf_to_tank_efficiency:
        The efficiency of the heat transfer process between the HTF in the heat
        exchanger immersed in the tank and the hot-water within the tank.

    :param previous_collector_output_temperature:
        The output temperature of HTF from the collector at the previous time step,
        measured in Kelvin.

    :param previous_tank_temperature:
        The tank temperature at the previous time step, measured in Kelvin.

    :return:
        An :class:`numpy.ndarray` containing the coefficients for the row in the matrix
        that represents the tank's htf equation.

    """

    tank_htf_equation_coefficients = numpy.zeros([1, 6])

    # Coefficients for the case where heat should be added to the collector.
    if previous_collector_output_temperature > previous_tank_temperature:
        # Collector input temperature term.
        tank_htf_equation_coefficients[0, 3] = -1
        # Collector output temperature term.
        tank_htf_equation_coefficients[0, 4] = 1 - htf_to_tank_efficiency
        # Tank temperature term.
        tank_htf_equation_coefficients[0, 5] = htf_to_tank_efficiency
    # Coefficients for the case where the fluid should pass straight through.
    else:
        # Collector input temperature term.
        tank_htf_equation_coefficients[0, 3] = -1
        # Collector output temperature term.
        tank_htf_equation_coefficients[0, 4] = 1

    return tank_htf_equation_coefficients


def _get_tank_equation_coefficients(
    current_hot_water_load: float,
    hot_water_tank: tank.Tank,
    htf_heat_capacity: float,
    htf_to_tank_efficiency: float,
    mass_flow_rate: float,
    previous_collector_output_temperature: float,
    previous_tank_temperature: float,
    resolution: int,
) -> numpy.ndarray:
    """
    Calculates the coefficient for the row representing the tank equation.

    :return:
        An :class:`numpy.ndarray` containing the coefficients for the row in the matrix
        that represents the tank equation.

    """

    tank_equation_coefficients = numpy.zeros([1, 6])

    heat_added_to_tank = (
        previous_collector_output_temperature > previous_tank_temperature
    )

    # Compute the output water temperature term.
    if heat_added_to_tank:
        tank_equation_coefficients[0, 4] = -(
            htf_to_tank_efficiency
            * mass_flow_rate  # [kg/s]
            * constants.HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        )  # [W/K]

    # Compute the tank-temperature term.
    tank_equation_coefficients[0, 5] = (
        # Change in internal energy
        hot_water_tank.mass  # [kg]
        * hot_water_tank.heat_capacity  # [J/kg*K]
        / resolution  # [s]
        # Heat addition
        + htf_to_tank_efficiency
        * mass_flow_rate  # [kg/s]
        * htf_heat_capacity  # [J/kg*K]
        * heat_added_to_tank
        # Tank heat loss
        + hot_water_tank.area  # [m^2]
        * hot_water_tank.heat_loss_coefficient  # [W/m^2*K]
        # Demand heat loss
        + current_hot_water_load  # [kg/s]
        * constants.HEAT_CAPACITY_OF_WATER  # [J/kg*K]
    )  # [W/K]

    return tank_equation_coefficients


def calculate_coefficient_matrix(
    collector_to_htf_efficiency: float,
    current_hot_water_load: float,
    hot_water_tank: tank.Tank,
    htf_to_tank_efficiency: float,
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,
) -> numpy.ndarray:
    """
    Calculates the matrix of coefficients required to solve the PV-T system itteratively

    :param collector_to_htf_efficiency:
        The efficiency of the heat transfer process between the thermal collector layer
        and the HTF passing through the collector.

    :param current_hot_water_load:
        The current hot water load, measured in kilograms (or litres) per second.

    :param hot_water_tank:
        A :class:`tank.Tank` instance representing the hot-water tank in the system.

    :param htf_to_tank_efficiency:
        The efficiency of the heat transfer process between the heat transfer fluid and
        the hot-water tank.

    :param previous_temperature_vector:
        An array containing the temperatures at the previous time step.

    :param pvt_panel:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The resolution of the model being run, measured in seconds.

    :param weather_conditions:
        The weather conditions at the current time step.

    :return:
        A :class:`numpy.ndarray` representing the matrix A in the above equation. All
        values have untis Watts per Kelvin.

    """

    # Unpack the temperature vector from the previous time step.
    (
        previous_glass_temperature,
        previous_pv_temperature,
        previous_collector_temperature,
        _,
        previous_collector_output_temperature,
        previous_tank_temperature,
    ) = previous_temperature_vector

    # Instantiate an empty array to represent the matrix.
    coefficient_matrix = numpy.zeros([6, 6])

    # Compute the glass-layer-equation coefficients.
    coefficient_matrix[0] = _get_glass_equation_coefficients(
        previous_collector_temperature,
        previous_glass_temperature,
        previous_pv_temperature,
        pvt_panel,
        resolution,
        weather_conditions,
    )

    # Compute the PV-layer-equation coefficients.
    coefficient_matrix[1] = _get_pv_equation_coefficients(
        previous_glass_temperature,
        previous_pv_temperature,
        pvt_panel,
        resolution,
    )

    # Compute the collector-layer-equation coefficients.
    coefficient_matrix[2] = _get_collector_equation_coefficients(
        collector_to_htf_efficiency,
        previous_collector_temperature,
        previous_glass_temperature,
        pvt_panel,
        resolution,
        weather_conditions,
    )

    coefficient_matrix[3] = _get_collector_htf_equation_coefficients(
        collector_to_htf_efficiency
    )

    coefficient_matrix[4] = _get_tank_htf_equation_coefficients(
        htf_to_tank_efficiency,
        previous_collector_output_temperature,
        previous_tank_temperature,
    )

    coefficient_matrix[5] = _get_tank_equation_coefficients(
        current_hot_water_load,
        hot_water_tank,
        pvt_panel.collector.htf_heat_capacity,
        htf_to_tank_efficiency,
        pvt_panel.collector.mass_flow_rate,
        previous_collector_output_temperature,
        previous_tank_temperature,
        resolution,
    )

    return coefficient_matrix


def calculate_resultant_vector(
    current_hot_water_load: float,
    hot_water_tank: tank.Tank,
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,
) -> numpy.ndarray:
    """
    Calculates the "resultant vector" required to solve the PV-T system itteratively.

    :param current_hot_water_load:
        The current hot-water load, measured in kilograms per second.

    :param hot_water_tank:
        A :class:`tank.Tank` instance representing the hot-water tank.

    :param previous_temperature_vector:
        An array containing the temperatures at the previous time step.

    :param pvt_panel:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The resolution of the model being run, measured in seconds.

    :param weather_conditions:
        The weather conditions at the current time step.

    :return:
        An array representing the resultant vector, I.E., the vector on the RHS of the
        matrix equation, denoted by `B` in the module docstring. All values have units
        Watts.

    """

    # Instantiate a vector to store the values computed and unpack the previous
    # temperatures
    resultant_vector = numpy.zeros([6, 1])
    (
        previous_glass_temperature,
        previous_pv_temperature,
        previous_collector_temperature,
        _,
        _,
        previous_tank_temperature,
    ) = previous_temperature_vector

    # Compute the glass-layer-equation value.
    resultant_vector[0] = (
        pvt_panel.glass.absorptivity
        * pvt_panel.area  # [m^2]
        * weather_conditions.irradiance  # [W]
        + weather_conditions.wind_heat_transfer_coefficient  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * weather_conditions.ambient_temperature  # [K]
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=previous_glass_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * weather_conditions.sky_temperature  # [K]
        + pvt_panel.glass.mass  # [kg]
        * pvt_panel.glass.heat_capacity  # [J/kg*K]
        * previous_glass_temperature  # [K]
        / resolution  # [s]
    )  # [W]

    # Compute the PV-layer-equation value.
    resultant_vector[1] = (
        physics_utils.transmissivity_absorptivity_product(
            diffuse_reflection_coefficient=pvt_panel.glass.diffuse_reflection_coefficient,
            glass_transmissivity=pvt_panel.glass.transmissivity,
            layer_absorptivity=pvt_panel.pv.absorptivity,
        )
        * weather_conditions.irradiance  # [W/m^2]
        * pvt_panel.pv.area  # [m^2]
        * (
            1
            - pvt_panel.pv.reference_efficiency
            * (
                1
                + pvt_panel.pv.thermal_coefficient * pvt_panel.pv.reference_temperature
            )
        )
        + pvt_panel.pv.mass
        * pvt_panel.pv.heat_capacity
        * previous_pv_temperature
        / resolution
    )

    # Compute the collector-layer-equation value.
    resultant_vector[2] = (
        # Solar heat input
        physics_utils.transmissivity_absorptivity_product(
            diffuse_reflection_coefficient=pvt_panel.glass.diffuse_reflection_coefficient,
            glass_transmissivity=pvt_panel.glass.transmissivity,
            layer_absorptivity=pvt_panel.collector.absorptivity,
        )
        * weather_conditions.irradiance  # [W/m^2]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Internal energy change.
        + pvt_panel.collector.mass  # [kg]
        * pvt_panel.collector.heat_capacity  # [J/kg*K]
        * previous_collector_temperature  # [K]
        * constants.NC
        / resolution  # [s]
        # Back plate heat loss.
        + pvt_panel.back_plate.conductance  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        * weather_conditions.ambient_temperature  # [K]
    )  # [W]

    resultant_vector[5] = (
        # Internal tank heat change.
        hot_water_tank.mass  # [kg]
        * hot_water_tank.heat_capacity  # [J/kg*K]
        * previous_tank_temperature  # [K]
        / resolution  # [s]
        # Tank heat loss.
        + hot_water_tank.area  # [m^2]
        * hot_water_tank.heat_loss_coefficient  # [W/m^2*K]
        * weather_conditions.ambient_temperature  # [K]
        # Demand water enthalpy gain.
        + current_hot_water_load  # [kg/s]
        * constants.HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        * weather_conditions.mains_water_temperature  # [K]
    )  # [W]

    return resultant_vector
