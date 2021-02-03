#!/usr/bin/python3.7
########################################################################################
# pvt_panel/test_utils.py - MUT for the utility module for the PVT panel.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The Module Unit Tests (MUT) for the PVT panel's utility module.

This module tests the various internal and external methods provided by the utility
module for the PVT panel.

"""

import unittest

from unittest import mock

import pytest

from ...__utils__ import (
    LayerParameters,
    OpticalLayerParameters,
)

from .. import __utils__

# The precision to be used in pytest tests.
PYTEST_PRECISION = 10 ** -6


class TestInstantiate(unittest.TestCase):
    """
    Tests the instantisation of the various dataclasses in the utility module.

    """

    def setUp(self) -> None:
        """
        Set up mocks and utilities in common across the test cases below.

        """

        super().setUp()
        self.layer_params = LayerParameters(
            mass=100, heat_capacity=200, area=15, thickness=0.001, temperature=273
        )
        self.optical_layer_params = OpticalLayerParameters(
            mass=self.layer_params.mass,
            heat_capacity=self.layer_params.heat_capacity,
            area=self.layer_params.area,
            thickness=self.layer_params.thickness,
            temperature=self.layer_params.temperature,
            transmissivity=0.9,
            absorptivity=0.8,
            emissivity=0.7,
        )

    def test_instantiate_layer(self) -> None:
        """
        Tests the successful instantisation of a :class:`__utils__.Layer` instance.

        """

        layer = __utils__.Layer(self.layer_params)

        self.assertEqual(
            self.layer_params.mass, layer._mass  # pylint: disable=protected-access
        )
        self.assertEqual(
            self.layer_params.heat_capacity,
            layer._heat_capacity,  # pylint: disable=protected-access
        )
        self.assertEqual(self.layer_params.area, layer.area)
        self.assertEqual(self.layer_params.thickness, layer.thickness)
        self.assertEqual(self.layer_params.temperature, layer.temperature)

    def test_instantiate_optical_layer(self) -> None:
        """
        Tests the successful instantisation of a :class:`__utils__.OpticalLayer`.

        """

        optical_layer = __utils__.OpticalLayer(self.optical_layer_params)

        self.assertEqual(
            self.optical_layer_params.absorptivity, optical_layer.absorptivity
        )
        self.assertEqual(self.optical_layer_params.emissivity, optical_layer.emissivity)
        self.assertEqual(
            self.optical_layer_params.transmissivity, optical_layer.transmissivity
        )


class TestPhysics(unittest.TestCase):
    """
    Tests the various Physics equations that are contained within the utility module.

    """

    def test_conductive_heat_transfer_no_gap(self) -> None:
        """
        Tests the publicised equation for the conductive heat transfer with no gap.

        :equation:
            heat_flow = conductance * area * temperature_difference
            W = W/m^2*K * m^2 * K

        :values:
            contact_area = 15 m^2
            destination_temperature = 200 K
            source_temperature = 300 K
            thermal_conductance = 500 W/m^2*K

        :result:
            heat_flow = 7.5 * 10^5 W

        :source:
            https://ctherm.com/resources/helpful-links-tools/thermalresistanceandconductivity/#:~:text=Thermal%20conductance%20is%20the%20time,in%20W%2Fm2%E2%8B%85K.  # pylint: disable=line-too-long

        """

        expected_heat_flow = 7.5 * 10 ** 5
        calculated_heat_flow = __utils__.conductive_heat_transfer_no_gap(
            contact_area=15,
            destination_temperature=200,
            source_temperature=300,
            thermal_conductance=500,
        )

        self.assertEqual(expected_heat_flow, calculated_heat_flow)

    def test_conductive_heat_transfer_with_gap(self) -> None:
        """
        Tests the publicised equation for the conductive heat transfer with a gap.

        :equation:
            heat_flow = conductivity * area * temperature_difference / air_gap_thickness
            W = W/m*K * m^2 * K / m

        :values:
            air_gap_thickness = 0.01 m
            contact_area = 15 m^2
            destination_temperature = 200 K
            source_temperature = 300 K
            thermal_conductivity_of_air = 100 W / m*K

        :result:
            heat_flow = 1.5 * 10^7 W

        :source:
            Herrando, M., Markides, C. N., & Hellgardt, K. (2014).
            A UK-based assessment of hybrid PV and solar-thermal systems for domestic
            heating and power: System performance. Applied Energy, 122, 288–309.
            https://doi.org/10.1016/j.apenergy.2014.01.061

        :notes:
            Unfortunately, Maria's paper makes an assumption that the heat flow is
            mostly by conduction across the air gap between the panel layers. This
            ignores any internal convection, but allows her to treat the layer as an
            insulating air layer.

            The equation checked here is the one from Maria's paper.

        """

        expected_heat_flow = 1.5 * 10 ** 7

        # Mock the thermal conductivity of air to be 100 W / m*K
        with mock.patch(
            "pvt_model.pvt_panel.__utils__.THERMAL_CONDUCTIVITY_OF_AIR", 100
        ):
            calculated_heat_flow = __utils__.conductive_heat_transfer_with_gap(
                air_gap_thickness=0.01,
                contact_area=15,
                destination_temperature=200,
                source_temperature=300,
            )

        self.assertEqual(expected_heat_flow, calculated_heat_flow)

    def convective_heat_transfer_heat_transfer_to_fluid(self) -> None:
        """
        Tests the publicised equation for the convective heat transfer to a fluid.

        :equation:
            heat_flow = heat_transfer_coefficient * area * (temperature_diff) ^ b
            W = W/m^2*K^b * m^2 * K^b

        :values:
            contact_area = 15 m^2
            fluid_temperature = 200 K
            heat_transfer_coefficient = 500 W / m^2 * K
            scaling_factor (b) = 1
            wall_temperature = 300 K

        :result:
            heat_flow = 7.5 * 10 ^ 5

        """

        expected_heat_flow = 7.5 * 10 ** 5
        calculated_heat_flow = __utils__.convective_heat_transfer_to_fluid(
            contact_area=15,
            convective_heat_transfer_coefficient=500,
            fluid_temperature=200,
            wall_temperature=300,
        )

        self.assertEqual(expected_heat_flow, calculated_heat_flow)

    def test_radiative_heat_transfer_between_bodies(self) -> None:
        """
        Tests the publicised equation for the radiative heat transfer between bodies.

        The method publicised can compute the radiative heat transfer between two
        parallel plates as well as that from a body to the sky/a sink at infinity.
        The radiative transfer between two plates is checked here.

        :equation:
            heat_flow = area * stefan_boltzman_constant * joint_emissivity * (
                t_source ^ 4 - t_dest ^ 4
            )
            W = m^2 * W/m^2*K^4 * K^4

            with the joint emissivity being given by
            joint_emissivity = 1 / (
                1 / source_emissivity + 1 / destination_emissivity - 1
            )

        :values:
            contact_area = 15
            destination_emissivity = 0.3
            destination_temperature = 200 K
            source_emissivity = 0.5
            source_temperature = 300 K
            stefan_boltzman_constant = 7 * 10 ^ -8

        :result:
            heat_flow = 1575 W

        :source:
            https://wiki.epfl.ch/me341-hmt/documents/lectures/slides_10_Radiation.pdf
            Equation 10.24.

        """

        expected_heat_flow = 1575

        # Mock the value of the Stefan Boltzman constant
        with mock.patch(
            "pvt_model.pvt_panel.__utils__.STEFAN_BOLTZMAN_CONSTANT", 7 * 10 ** (-8)
        ):
            calculated_heat_flow = __utils__.radiative_heat_transfer(
                destination_emissivity=0.3,
                destination_temperature=200,
                radiative_contact_area=15,
                source_emissivity=0.5,
                source_temperature=300,
            )

        self.assertEqual(
            pytest.approx(expected_heat_flow, PYTEST_PRECISION),
            pytest.approx(calculated_heat_flow, PYTEST_PRECISION),
        )

    def test_radiative_heat_transfer_to_sky(self) -> None:
        """
        Tests the publicised equation for the radiative heat transfer between bodies.

        The method publicised can compute the radiative heat transfer between two
        parallel plates as well as that from a body to the sky/a sink at infinity.
        The radiative transfer to the sky is checked here.

        :equation:
            heat_flow = area * stefan_boltzman_constant * emissivity * (
                t_source ^ 4 - t_dest ^ 4
            )
            W = m^2 * W/m^2*K^4 * K^4

        :values:
            contact_area = 15
            destination_temperature = 3 K
            source_emissivity = 0.5
            source_temperature = 373 K
            stefan_boltzman_constant = 7 * 10 ^ -8

        :result:
            heat_flow = 10162.361244 W

        """

        expected_heat_flow = 10162.361244

        # Mock the value of the Stefan Boltzman constant
        with mock.patch(
            "pvt_model.pvt_panel.__utils__.STEFAN_BOLTZMAN_CONSTANT", 7 * 10 ** (-8)
        ):
            calculated_heat_flow = __utils__.radiative_heat_transfer(
                destination_temperature=3,
                radiating_to_sky=True,
                radiative_contact_area=15,
                source_emissivity=0.5,
                source_temperature=373,
            )

        self.assertEqual(
            pytest.approx(expected_heat_flow, PYTEST_PRECISION),
            pytest.approx(calculated_heat_flow, PYTEST_PRECISION),
        )

    def test_solar_heat_input_electrical_layer(self) -> None:
        """
        Tests the publicised equation for calculating the solar heat input for PV layers

        :equation:
            solar_heat_input = solar_irradiance * area * ta_product * (
                1 - electrical_efficiency
            )
            W = W/m^2 * m^2

            where the ta product is specific to the layer and calculated elsewhere.

        :values:
            area = 15 m^2
            electrical_efficiency = 0.4
            solar_irradiance = 500 W / m^2
            ta_product = 0.72

        :result:
            solar_heat_input = 3240 W

        """

        expected_solar_heat_input = 3240

        calculated_solar_heat_input = __utils__.solar_heat_input(
            area=15,
            electrical_efficiency=0.4,
            solar_energy_input=500,
            ta_product=0.72,
        )

        self.assertEqual(
            pytest.approx(expected_solar_heat_input, PYTEST_PRECISION),
            pytest.approx(calculated_solar_heat_input, PYTEST_PRECISION),
        )

    def test_solar_heat_input_non_electrical_layer(self) -> None:
        """
        Tests the equation for calculating the solar heat for non-electric layers.

        :equation:
            solar_heat_input = solar_irradiance * area * ta_product
            W = W/m^2 * m^2

            where the ta product is specific to the layer and calculated elsewhere.

        :values:
            area = 15 m^2
            solar_irradiance = 500 W / m^2
            ta_product = 0.72

        :result:
            solar_heat_input = 3240 W

        """

        expected_solar_heat_input = 5400

        calculated_solar_heat_input = __utils__.solar_heat_input(
            area=15,
            solar_energy_input=500,
            ta_product=0.72,
        )

        self.assertEqual(
            pytest.approx(expected_solar_heat_input, PYTEST_PRECISION),
            pytest.approx(calculated_solar_heat_input, PYTEST_PRECISION),
        )

    def test_transmissivity_absorptivity_product(self) -> None:
        """
        Tests the publicised equation for the transmissivity absorptivity product.

        :equation:
            ta_product = (
                (glass_transmissivity * layer_absorptivity)
                / (1 - (1 - layer_absorptivity) * diffuse_reflectivity_coefficient)
            )

        :values:
            glass_transmissivity = 0.9
            diffuse_reflectivity_coefficient = 0.5
            layer_absorptivity = 0.8

        :result:
            ta_product = 0.8

        """

        expected_ta_product = 0.8

        calculated_ta_product = __utils__.transmissivity_absorptivity_product(
            diffuse_reflection_coefficient=0.5,
            glass_transmissivity=0.9,
            layer_absorptivity=0.8,
        )

        self.assertEqual(
            pytest.approx(expected_ta_product, PYTEST_PRECISION),
            pytest.approx(calculated_ta_product, PYTEST_PRECISION),
        )

    def test_wind_heat_transfer(self) -> None:
        """
        Tests the publicised equation for the wind heat transfer.

        :equation:
            heat_loss = convective_heat_transfer_coefficient * area * (
                source_temperature - air_temperature
            )

        :values:
            area = 15 m^2
            convective_heat_transfer_coefficient = 5 W / K
            destination_tempreature = 300 K
            source_temperature = 270 K

        :result:
            heat_loss = 2250 W

        """

        expected_heat_loss = 2250

        calculated_heat_loss = __utils__.wind_heat_transfer(
            contact_area=15,
            destination_temperature=270,
            source_temperature=300,
            wind_heat_transfer_coefficient=5,
        )

        self.assertEqual(
            pytest.approx(expected_heat_loss, PYTEST_PRECISION),
            pytest.approx(calculated_heat_loss, PYTEST_PRECISION),
        )
