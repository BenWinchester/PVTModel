#!/usr/bin/python3.7
########################################################################################
# mains_power.py - Represents a mains power source, such as a grid or pipeline network.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The mains power module for this PV-T system model.

This module represents a mains grid, and contains code and information pertaining to a
mains power grid and gas supply. Among other things, the module is designed with the
goal of computing carbon dioxide outputs generated and mitigating by using the PV-T
system being modelled.

"""

from dataclasses import dataclass
from typing import Any

from ..__utils__ import (
    CarbonEmissions,
    read_yaml,
    TotalPowerData,
)

from .__utils__ import UtilityType


@dataclass
class _Utility:
    """
    Represents a utility, such as electricity or gas.

    .. attribute:: utility_type
        The type of the utility.

    .. attribute:: emissions_per_joule
        The emissions per joule of energy used, measured in kilograms of CO2 equivalent.

    """

    utility_type: UtilityType
    emissions_per_joule: float

    def __repr__(self) -> str:
        """
        Returns a nice-looking representation of the :class:`Utility` instance.

        :return:
            A `str` giving a nice-looking representation of the utility.

        """

        return (
            "Utility("
            f"emissions_per_joule={self.emissions_per_joule}, "
            f"utility_type={self.utility_type}"
            ")"
        )


class MainsSupply:
    """
    Represents a mains power supply.

    """

    # Private Attributes:
    #
    # .. attribute:: _utilities
    #   A mapping of utility type to utility supplied by the mains supply.
    #

    def __init__(self, utilities) -> None:
        """
        Instantiate a mains supply.

        """

        self._utilities = utilities

    def __repr__(self) -> str:
        """
        Returns a nice-looking representation of the mains-power instance.

        :return:
            A `str` giving a representation of the class.

        """

        return f"MainsSupply(utilities={list(self._utilities.values())})"

    @classmethod
    def from_yaml(cls, yaml_data_path: str) -> Any:
        """
        Instantiate a :class:`MainsSupply` instance based on mains supply YAML data.

        :param yaml_data_path:
            The path to the yaml data file containing information about the mains supply

        :return:
            A :class:`MainsSupply` instance based on the YAML data

        """

        yaml_data = read_yaml(yaml_data_path)

        utilities = {
            UtilityType.electricity: _Utility(
                UtilityType.electricity, yaml_data["electricity"]
            ),
            UtilityType.gas: _Utility(UtilityType.gas, yaml_data["natural_gas"]),
        }

        return cls(utilities)

    def get_carbon_emissions(self, total_power_data: TotalPowerData) -> CarbonEmissions:
        """
        Computes the total amount of carbon dioxide produced in the run.

        :param total_power_data:
            Information about the total amount of power produced in the system.

        :return:
            Information about the carbon emissions.

        """

        electrical_carbon_produced = (
            total_power_data.electricity_demand - total_power_data.electricity_supplied
        ) * self._utilities[UtilityType.electricity].emissions_per_joule
        electrical_carbon_saved = (
            total_power_data.electricity_supplied
            * self._utilities[UtilityType.electricity].emissions_per_joule
        )
        heating_carbon_produced = (
            total_power_data.heating_demand - total_power_data.heating_supplied
        ) * self._utilities[UtilityType.gas].emissions_per_joule
        heating_carbon_saved = (
            total_power_data.heating_supplied
            * self._utilities[UtilityType.gas].emissions_per_joule
        )

        return CarbonEmissions(
            electrical_carbon_produced,
            electrical_carbon_saved,
            heating_carbon_produced,
            heating_carbon_saved,
        )
