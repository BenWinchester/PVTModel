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
from .__utils__ import TemperatureName, WeatherConditions

__all__ = (
    "calculate_coefficient_matrix",
    "calculate_resultant_vector",
)


def _get_glass_equation_coefficients(
    best_guess_collector_temperature: float,
    best_guess_glass_temperature: float,
    previous_pv_temperature: float,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,
) -> numpy.ndarray:
    """
    Calculates the coefficient for the row representing the glass-layer equation.

    :param best_guess_collector_temperature:
        The best guess for the temperature of the collector layer at the time step being
        calculated, measured in Kelvin.

    :param best_guess_glass_temperature:
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
    glass_equation_coefficients = numpy.zeros([1, len(TemperatureName)])

    # Compute the glass temperature term.
    glass_equation_coefficients[0][TemperatureName.glass.value] = (
        # Radiative heat transfer to the sky
        physics_utils.radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=best_guess_glass_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        # Conductive heat transfer to the wind
        + weather_conditions.wind_heat_transfer_coefficient  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        # Radiative heat transfer with the PV layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=best_guess_glass_temperature,
            source_emissivity=pvt_panel.pv.emissivity,
            source_temperature=previous_pv_temperature,
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Radiative heat transfer from the collector layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=best_guess_glass_temperature,
            source_emissivity=pvt_panel.collector.emissivity,
            source_temperature=best_guess_collector_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Conductive heat transfer with the PV and collector layers
        + physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        # Change in internal energy of the glass layer
        # + (
        #     pvt_panel.glass.mass  # [kg]
        #     * pvt_panel.glass.heat_capacity  # [J/kg*K]
        #     / resolution  # [s]
        # )  # [W/s]
    )  # [W/K]

    # Compute the PV temperature term.
    glass_equation_coefficients[0][TemperatureName.pv.value] = -(
        # Conductive heat transfer from the PV layer
        physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Radiative heat transfer from the PV layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=best_guess_glass_temperature,
            source_emissivity=pvt_panel.pv.emissivity,
            source_temperature=previous_pv_temperature,
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
    )  # [W/K]

    # Compute the collector temperature term.
    glass_equation_coefficients[0][TemperatureName.collector.value] = -(
        # Conductive heat transfer from the collector layer
        physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Radiative heat transfer from the collector layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=best_guess_glass_temperature,
            source_emissivity=pvt_panel.collector.emissivity,
            source_temperature=best_guess_collector_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
    )  # [W/K]

    return glass_equation_coefficients


def _get_pv_equation_coefficients(
    best_guess_glass_temperature: float,
    best_guess_pv_temperature: float,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,
) -> numpy.ndarray:
    """
    Calculates the coefficient for the row representing the PV-layer equation.

    :param best_guess_glass_temperature:
        The temperature of the glass layer at the previous time step, measured in
        Kelvin.

    :param best_guess_pv_temperature:
        The best guess for the temperature of the PV layer at the current time step,
        measured in Kelvin.

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

    pv_equation_coefficients = numpy.zeros([1, len(TemperatureName)])

    # Compute the glass temperature term.
    pv_equation_coefficients[0][TemperatureName.glass.value] = -(
        # Conductive heat transfer from the PV layer
        physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Radiative heat transfer from the PV layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.pv.emissivity,
            destination_temperature=best_guess_pv_temperature,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=best_guess_glass_temperature,
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
    )  # [W/K]

    # Compute the PV temperature term.
    pv_equation_coefficients[0][TemperatureName.pv.value] = (
        # Conductive heat transfer from the glass layer.
        physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Radiative heat transfer from the glass layer.
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.pv.emissivity,
            destination_temperature=best_guess_pv_temperature,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=best_guess_glass_temperature,
        )  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Conductive heat transfer to the collector layer.
        + pvt_panel.pv_to_collector_thermal_conductance  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
        # Change in internal energy of the PV layer.
        # + pvt_panel.pv.mass  # [kg]
        # * pvt_panel.pv.heat_capacity  # [J/kg*K]
        # * constants.NUMBER_OF_COLLECTORS
        # / resolution  # [s]
        # Solar heat input.
        - physics_utils.transmissivity_absorptivity_product(
            diffuse_reflection_coefficient=pvt_panel.glass.diffuse_reflection_coefficient,
            glass_transmissivity=pvt_panel.glass.transmissivity,
            layer_absorptivity=pvt_panel.pv.absorptivity,
        )
        * pvt_panel.pv.area  # [m^2]
        * weather_conditions.irradiance  # [W/m^2]
        * pvt_panel.pv.reference_efficiency
        * pvt_panel.pv.thermal_coefficient  # [1/K]
    )  # [W/K]

    # Compute the collector temperature term.
    pv_equation_coefficients[0][TemperatureName.collector.value] = -(
        # Conductive heat transfer to the collector layer.
        pvt_panel.pv_to_collector_thermal_conductance  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
    )

    return pv_equation_coefficients


def _get_collector_equation_coefficients(
    collector_to_htf_efficiency: float,
    best_guess_collector_temperature: float,
    best_guess_glass_temperature: float,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,  # pylint: disable=unused-argument
) -> numpy.ndarray:
    """
    Calculates the coefficient for the row representing the collector-layer equation.

    :param collector_to_htf_efficiency:
        The efficiency of the heat transfer process between the thermal collector layer
        and the HTF in the collector tubes.

    :param best_guess_collector_temperature:
        The temperature of the collector layer at the previous time step, measured in
        Kelvin.

    :param best_guess_glass_temperature:
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

    collector_equation_coefficients = numpy.zeros([1, len(TemperatureName)])

    # Compute the glass temperature term.
    collector_equation_coefficients[0][TemperatureName.glass.value] = -(
        # Conductive heat transfer from the glass layer
        physics_utils.conductive_heat_transfer_coefficient_with_gap(
            pvt_panel.air_gap_thickness
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        # Radiative heat transfer from the glass layer
        + physics_utils.radiative_heat_transfer_coefficient(
            destination_emissivity=pvt_panel.glass.emissivity,
            destination_temperature=best_guess_glass_temperature,
            source_emissivity=pvt_panel.collector.emissivity,
            source_temperature=best_guess_collector_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
    )

    # Compute the PV temperature term.
    collector_equation_coefficients[0][TemperatureName.pv.value] = -(
        # Conductive heat transfer to the PV layer.
        pvt_panel.pv_to_collector_thermal_conductance  # [W/m^2*K]
        * pvt_panel.pv.area  # [m^2]
    )  # [W/K]

    # Compute the collector temperature term.
    collector_equation_coefficients[0][TemperatureName.collector.value] = (
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
            destination_temperature=best_guess_glass_temperature,
            source_emissivity=pvt_panel.collector.emissivity,
            source_temperature=best_guess_collector_temperature,
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
        # + (
        #     pvt_panel.collector.mass  # [kg]
        #     * pvt_panel.collector.heat_capacity  # [J/kg*K]
        #     + pvt_panel.back_plate.mass  # [kg]
        #     * pvt_panel.back_plate.heat_capacity  # [J/kg*K]
        # )
        # * constants.NUMBER_OF_COLLECTORS
        # / resolution  # [s]
    )  # [W/K]

    # Compute the collector input temperature term.
    # collector_equation_coefficients[0, 3] = -(
    #     collector_to_htf_efficiency
    #     * pvt_panel.collector.mass_flow_rate  # [kg/s]
    #     * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
    # )  # [W/K]

    return collector_equation_coefficients


def _get_bulk_water_equation_coefficients() -> numpy.ndarray:
    """
    Calculates the coefficients for the equation for the bulk water

    :return:
        An array containing these coefficients.

    """

    collector_to_tank_pipe_equation_coefficients = numpy.zeros(
        [1, len(TemperatureName)]
    )

    collector_to_tank_pipe_equation_coefficients[0][
        TemperatureName.bulk_water.value
    ] = 1
    collector_to_tank_pipe_equation_coefficients[0][
        TemperatureName.collector_input.value
    ] = -0.5
    collector_to_tank_pipe_equation_coefficients[0][
        TemperatureName.collector_output.value
    ] = -0.5

    return collector_to_tank_pipe_equation_coefficients


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

    collector_htf_equation_coefficients = numpy.zeros([1, len(TemperatureName)])

    # Collector temperature term.
    collector_htf_equation_coefficients[0][
        TemperatureName.collector.value
    ] = collector_to_htf_efficiency
    # Collector input temperature term.
    collector_htf_equation_coefficients[0][TemperatureName.collector_input.value] = (
        1 - collector_to_htf_efficiency
    )
    # Collector output temperature term.
    collector_htf_equation_coefficients[0][TemperatureName.collector_output.value] = -1

    return collector_htf_equation_coefficients


def _get_collector_to_tank_pipe_equation() -> numpy.ndarray:
    """
    Calculates the coefficients for the equation from the collector to the tank.

    :return:
        An array containing these coefficients.

    """

    collector_to_tank_pipe_equation_coefficients = numpy.zeros(
        [1, len(TemperatureName)]
    )

    collector_to_tank_pipe_equation_coefficients[0][
        TemperatureName.collector_output.value
    ] = 1
    collector_to_tank_pipe_equation_coefficients[0][
        TemperatureName.tank_input.value
    ] = -1

    return collector_to_tank_pipe_equation_coefficients


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

    tank_htf_equation_coefficients = numpy.zeros([1, len(TemperatureName)])

    # Coefficients for the case where heat should be added to the collector.
    if previous_collector_output_temperature > previous_tank_temperature:
        # Collector input temperature term.
        tank_htf_equation_coefficients[0][TemperatureName.tank_output.value] = -1
        # Collector output temperature term.
        tank_htf_equation_coefficients[0][TemperatureName.tank_input.value] = (
            1 - htf_to_tank_efficiency
        )
        # Tank temperature term.
        tank_htf_equation_coefficients[0][
            TemperatureName.tank.value
        ] = htf_to_tank_efficiency
    # Coefficients for the case where the fluid should pass straight through.
    else:
        # Collector input temperature term.
        tank_htf_equation_coefficients[0][TemperatureName.tank_output.value] = -1
        # Collector output temperature term.
        tank_htf_equation_coefficients[0][TemperatureName.tank_input.value] = 1

    return tank_htf_equation_coefficients


def _get_tank_to_collector_pipe_equation_coefficients() -> numpy.ndarray:
    """
    Calculates the coefficients for the equation from the collector to the tank.

    :return:
        An array containing these coefficients.

    """

    tank_to_collector_pipe_equation_coefficients = numpy.zeros(
        [1, len(TemperatureName)]
    )

    tank_to_collector_pipe_equation_coefficients[0][
        TemperatureName.collector_input.value
    ] = 1
    tank_to_collector_pipe_equation_coefficients[0][
        TemperatureName.tank_output.value
    ] = -1

    return tank_to_collector_pipe_equation_coefficients


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

    tank_equation_coefficients = numpy.zeros([1, len(TemperatureName)])

    heat_added_to_tank = (
        previous_collector_output_temperature > previous_tank_temperature + 1
    )

    # Compute the output water temperature term.
    if heat_added_to_tank:
        tank_equation_coefficients[0][TemperatureName.tank_input.value] = -(
            htf_to_tank_efficiency
            * mass_flow_rate  # [kg/s]
            * constants.HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        )  # [W/K]

    # Compute the tank-temperature term.
    tank_equation_coefficients[0][TemperatureName.tank.value] = (
        # Heat addition
        htf_to_tank_efficiency
        * mass_flow_rate  # [kg/s]
        * htf_heat_capacity  # [J/kg*K]
        * heat_added_to_tank
        # Tank heat loss
        + hot_water_tank.area  # [m^2]
        * hot_water_tank.heat_loss_coefficient  # [W/m^2*K]
        # Demand heat loss
        + current_hot_water_load  # [kg/s]
        * constants.HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        # Change in internal energy
        + hot_water_tank.mass  # [kg]
        * hot_water_tank.heat_capacity  # [J/kg*K]
        / resolution  # [s]
    )  # [W/K]

    return tank_equation_coefficients


def calculate_coefficient_matrix(
    best_guess_temperature_vector: numpy.ndarray,
    collector_to_htf_efficiency: float,
    current_hot_water_load: float,
    hot_water_tank: tank.Tank,
    htf_to_tank_efficiency: float,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,
) -> numpy.ndarray:
    """
    Calculates the matrix of coefficients required to solve the PV-T system itteratively

    :param best_guess_temperature_vector:
        An array containing the best guess of temperatures at the next time step, i.e.,
        the time step being computed. The radiative heat transfer, for instance, depends
        in a non-linear way on the temperatures at the next time step. In order to
        estimate these values well, a best guess is needed.

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

    # Unpack the temperature vectors for the best guess time step and previous time step
    best_guess_glass_temperature = best_guess_temperature_vector[
        TemperatureName.glass.value
    ]
    best_guess_pv_temperature = best_guess_temperature_vector[TemperatureName.pv.value]
    best_guess_collector_temperature = best_guess_temperature_vector[
        TemperatureName.collector.value
    ]
    best_guess_collector_output_temperature = best_guess_temperature_vector[
        TemperatureName.collector_output.value
    ]
    best_guess_tank_temperature = best_guess_temperature_vector[
        TemperatureName.tank.value
    ]

    # Instantiate an empty array to represent the matrix.
    coefficient_matrix = numpy.zeros([len(TemperatureName), len(TemperatureName)])

    # Compute the glass-layer-equation coefficients.
    coefficient_matrix[TemperatureName.glass.value] = _get_glass_equation_coefficients(
        best_guess_collector_temperature,
        best_guess_glass_temperature,
        best_guess_pv_temperature,
        pvt_panel,
        resolution,
        weather_conditions,
    )

    # Compute the PV-layer-equation coefficients.
    coefficient_matrix[TemperatureName.pv.value] = _get_pv_equation_coefficients(
        best_guess_glass_temperature,
        best_guess_pv_temperature,
        pvt_panel,
        resolution,
        weather_conditions,
    )

    # Compute the collector-layer-equation coefficients.
    coefficient_matrix[
        TemperatureName.collector.value
    ] = _get_collector_equation_coefficients(
        collector_to_htf_efficiency,
        best_guess_collector_temperature,
        best_guess_glass_temperature,
        pvt_panel,
        resolution,
        weather_conditions,
    )

    coefficient_matrix[
        TemperatureName.bulk_water.value
    ] = _get_bulk_water_equation_coefficients()

    coefficient_matrix[
        TemperatureName.collector_input.value
    ] = _get_collector_htf_equation_coefficients(collector_to_htf_efficiency)

    coefficient_matrix[
        TemperatureName.collector_output.value
    ] = _get_tank_htf_equation_coefficients(
        htf_to_tank_efficiency,
        best_guess_collector_output_temperature,
        best_guess_tank_temperature,
    )

    coefficient_matrix[
        TemperatureName.tank_input.value
    ] = _get_collector_to_tank_pipe_equation()

    coefficient_matrix[TemperatureName.tank.value] = _get_tank_equation_coefficients(
        current_hot_water_load,
        hot_water_tank,
        pvt_panel.collector.htf_heat_capacity,
        htf_to_tank_efficiency,
        pvt_panel.collector.mass_flow_rate,
        best_guess_collector_output_temperature,
        best_guess_tank_temperature,
        resolution,
    )

    coefficient_matrix[
        TemperatureName.tank_output.value
    ] = _get_tank_to_collector_pipe_equation_coefficients()

    return coefficient_matrix


def calculate_resultant_vector(
    best_guess_glass_temperature: float,
    collector_to_htf_efficiency: float,
    current_hot_water_load: float,
    hot_water_tank: tank.Tank,
    previous_temperature_vector: numpy.ndarray,
    pvt_panel: pvt.PVT,
    resolution: int,
    weather_conditions: WeatherConditions,
) -> numpy.ndarray:
    """
    Calculates the "resultant vector" required to solve the PV-T system itteratively.

    :best_guess_glass_temperature:
        The best guess for the temperature of the glass layer at the current time step.

    :param collector_to_htf_efficiency:
        The efficiency of the heat transfer process between the thermal collector layer
        and the HTF passing through the collector.

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
    resultant_vector = numpy.zeros([len(TemperatureName), 1])
    previous_collector_input_temperature = previous_temperature_vector[
        TemperatureName.collector_input.value
    ]
    previous_tank_temperature = previous_temperature_vector[TemperatureName.tank.value]

    # Compute the glass-layer-equation value.
    resultant_vector[TemperatureName.glass.value] = (
        # Radiative heat transfer to the sky
        physics_utils.radiative_heat_transfer_coefficient(
            destination_temperature=weather_conditions.sky_temperature,
            radiating_to_sky=True,
            source_emissivity=pvt_panel.glass.emissivity,
            source_temperature=best_guess_glass_temperature,
        )  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * weather_conditions.sky_temperature  # [K]
        # Convective heat loss to the wind.
        + weather_conditions.wind_heat_transfer_coefficient  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * weather_conditions.ambient_temperature  # [K]
        # Solar absorption
        # + pvt_panel.glass.absorptivity
        # * pvt_panel.area  # [m^2]
        # * weather_conditions.irradiance  # [W]
        # Change in internal energy
        # + pvt_panel.glass.mass  # [kg]
        # * pvt_panel.glass.heat_capacity  # [J/kg*K]
        # * previous_glass_temperature  # [K]
        # / resolution  # [s]
    )  # [W]

    # Compute the PV-layer-equation value.
    resultant_vector[TemperatureName.pv.value] = (
        # Solar heat input
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
        )  # [W]
        # Change in internal energy.
        # + pvt_panel.pv.mass  # [kg]
        # * pvt_panel.pv.heat_capacity  # [J/kg*K]
        # * previous_pv_temperature  # [K]
        # * constants.NUMBER_OF_COLLECTORS
        # / resolution  # [s]
    )  # [W]

    # Compute the collector-layer-equation value.
    resultant_vector[TemperatureName.collector.value] = (
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
        # + (
        #     pvt_panel.collector.mass  # [kg]
        #     * pvt_panel.collector.heat_capacity  # [J/kg*K]
        #     + pvt_panel.back_plate.mass  # [kg]
        #     * pvt_panel.back_plate.heat_capacity  # [J/kg*K]
        # )
        # * previous_collector_temperature  # [K]
        # * constants.NUMBER_OF_COLLECTORS
        # / resolution  # [s]
        # Back plate heat loss.
        + pvt_panel.back_plate.conductance  # [W/m^2*K]
        * pvt_panel.area  # [m^2]
        * (1 - pvt_panel.portion_covered)
        * weather_conditions.ambient_temperature  # [K]
        # Compute the collector-input-htf temperature term.
        + collector_to_htf_efficiency
        * pvt_panel.collector.mass_flow_rate  # [kg/s]
        * pvt_panel.collector.htf_heat_capacity  # [J/kg*K]
        * previous_collector_input_temperature  # [K]
    )  # [W]

    # resultant_vector[TemperatureName.collector_input.value] = (
    #     collector_to_htf_efficiency - 1
    # ) * previous_collector_input_temperature  # [K]

    resultant_vector[TemperatureName.tank.value] = (
        # Tank heat loss.
        # 573
        hot_water_tank.area  # [m^2]
        * hot_water_tank.heat_loss_coefficient  # [W/m^2*K]
        * weather_conditions.ambient_temperature  # [K]
        # Demand water enthalpy gain.
        + current_hot_water_load  # [kg/s]
        * constants.HEAT_CAPACITY_OF_WATER  # [J/kg*K]
        * weather_conditions.mains_water_temperature  # [K]
        # Internal tank heat change.
        + hot_water_tank.mass  # [kg]
        * hot_water_tank.heat_capacity  # [J/kg*K]
        * previous_tank_temperature  # [K]
        / resolution  # [s]
    )  # [W]

    return resultant_vector
