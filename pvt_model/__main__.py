#!/usr/bin/python3.7
########################################################################################
# __main__.py - The main module for the higher-level PV-T model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The main module for the PV-T model.

The model is run from here for several runs as determined by command-line arguments.

"""

import sys

from typing import Dict

from . import argparser
from .pvt_system_model import argparser as pvt_system_argparser
from .pvt_system_model import tank
from .pvt_system_model.pvt_panel import pvt

from .pvt_system_model.__main__ import main as pvt_system_model_main
from .pvt_system_model.__utils__ import TemperatureName
from .pvt_system_model.constants import (
    DENSITY_OF_WATER,
    HEAT_CAPACITY_OF_WATER,
    THERMAL_CONDUCTIVITY_OF_WATER,
    INITIAL_SYSTEM_TEMPERATURE_MAPPING,
)
from .pvt_system_model.process_pvt_system_data import (
    hot_water_tank_from_path,
    pvt_panel_from_path,
)

from .__utils__ import fourier_number, get_logger, LOGGER_NAME


def _get_system_fourier_numbers(
    hot_water_tank: tank.Tank, pvt_panel: pvt.PVT, resolution: float
) -> Dict[TemperatureName, float]:
    """
    Determine the Fourier numbers of the various system components.

    :param hot_water_tank:
        A :class:`tank.Tank` instance representing the hot-water tank in the system.

    :param pvt_panel:
        A :class:`pvt.PVT` instance representing the PVT panel being modelled.

    :param resolution:
        The resolution being used for the model.

    :return:
        The Fourier coefficients for the various components being modelled.

    """

    # Determine the Fourier coefficients of the panel's layers.
    fourier_number_map: Dict[TemperatureName, float] = dict()
    fourier_number_map[TemperatureName.glass] = fourier_number(
        pvt_panel.glass.thickness,
        pvt_panel.glass.conductivity,
        pvt_panel.glass.density,
        pvt_panel.glass.heat_capacity,
        resolution,
    )
    fourier_number_map[TemperatureName.collector] = fourier_number(
        pvt_panel.collector.thickness,
        pvt_panel.collector.conductivity,
        pvt_panel.collector.density,
        pvt_panel.collector.heat_capacity,
        resolution,
    )
    fourier_number_map[TemperatureName.pv] = fourier_number(
        pvt_panel.pv.thickness,
        pvt_panel.pv.conductivity,
        pvt_panel.pv.density,
        pvt_panel.pv.heat_capacity,
        resolution,
    )
    fourier_number_map[TemperatureName.bulk_water] = fourier_number(
        pvt_panel.collector.pipe_diameter,
        THERMAL_CONDUCTIVITY_OF_WATER,
        DENSITY_OF_WATER,
        pvt_panel.collector.htf_heat_capacity,
        resolution,
    )
    fourier_number_map[TemperatureName.tank] = fourier_number(
        hot_water_tank.diameter,
        THERMAL_CONDUCTIVITY_OF_WATER,
        DENSITY_OF_WATER,
        HEAT_CAPACITY_OF_WATER,
        resolution,
    )

    return fourier_number_map


def main(args) -> None:
    """
    The main module for the code.

    :param args:
        The command-line arguments passed into the component.

    """

    # Initialise logging.
    logger = get_logger(LOGGER_NAME)
    logger.info(
        "%s PVT model instantiated. %s\nCommand: %s", "=" * 20, "=" * 20, " ".join(args)
    )
    print("PVT model instantiated.")

    # Parse the arguments passed in.
    parsed_args, unknown_args = argparser.parse_args(args)

    # Determine the Fourier number for the PV-T panel.
    if parsed_args.dynamic or parsed_args.quasi_steady:
        logger.info("Skipping Fourier number calculation.")
    else:
        logger.info("Beginning Fourier number calculation.")
        # Determine the arguments needed for instantiating the PVT system model.
        parsed_pvt_system_args = pvt_system_argparser.parse_args(unknown_args)

        # Instantiate a PVT panel instance based on the data.
        pvt_panel = pvt_panel_from_path(
            INITIAL_SYSTEM_TEMPERATURE_MAPPING[TemperatureName.bulk_water],
            parsed_pvt_system_args.portion_covered,
            parsed_pvt_system_args.pvt_data_file,
            parsed_pvt_system_args.unglazed,
        )

        # Instantiate a hot-water tank instance based on the data.
        hot_water_tank = hot_water_tank_from_path(parsed_pvt_system_args.tank_data_file)

        # Determine the Fourier numbers.
        fourier_number_map = _get_system_fourier_numbers(
            hot_water_tank, pvt_panel, parsed_pvt_system_args.resolution
        )

        logger.info(
            "Fourier numbers determined:\n%s",
            "\n".join(
                [
                    f"Fourier number of {key.name}: {value}"
                    for key, value in fourier_number_map.items()
                ]
            ),
        )
        print(
            "Fourier numbers determined:\n{}".format(
                "\n".join(
                    [
                        f"Fourier number of {key.name}: {value}"
                        for key, value in fourier_number_map.items()
                    ]
                )
            )
        )

    # Determine the initial conditions for the run via iteration until the initial and
    # final temperatures for the day match up.
    import pdb

    pdb.set_trace()
    initial_system_temperature_vector = [
        INITIAL_SYSTEM_TEMPERATURE_MAPPING[temperature_name]
        for temperature_name in sorted(TemperatureName, key=lambda entry: entry.value)
    ]
    unknown_args.append("--return-system-data")
    unknown_args.extend(
        ["--initial-system-temperature-vector"].extend(
            initial_system_temperature_vector
        )
    )
    system_data = pvt_system_model_main(unknown_args)


if __name__ == "__main__":
    main(sys.argv[1:])
