#!/usr/bin/python3.7
########################################################################################
# argparser.py - The argument parser module for the PVT model.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The argument parser module for the PV-T model.

This module contains argument parsing code for the PVT module.

In a normal component, the argument parsing would happen within the main module.
However, the large number of arguments makes it neater for the code to be in a separate
module for the model.

"""

import argparse

from logging import Logger
from typing import Set

from .__utils__ import BColours, TemperatureName

__all__ = ("check_args", "parse_args")

# Used to keep track of the temperature layers based on the CLI arguments passsed in.
layer_map = {
    "g": TemperatureName.glass,
    "pv": TemperatureName.pv,
    "a": TemperatureName.absorber,
    "p": TemperatureName.pipe,
    "f": TemperatureName.htf,
}


class ArgumentMismatchError(Exception):
    """
    Raised when arguments passed in mismatch.

    """

    def __init__(self, msg: str) -> None:
        """
        Instantiate a :class:`ArgumentMismatchError`.

        :param msg:
            The message to append to the output.

        """

        super().__init__(f"Mismatch in command-line arguments: {msg}")


def check_args(
    parsed_args: argparse.Namespace, logger: Logger, number_of_pipes: int
) -> Set[TemperatureName]:
    """
    Enforces rules on the command-line arguments passed in in addition to argparse rules

    :param parsed_args:
        The parsed command-line arguments.

    :param logger:
        The logger used for the run.

    :param number_of_pipes:
        The number of pipes attached to the absorber being modelled.

    :raises: ArgumentMismatchError
        Raised if the command-line arguments mismatch.

    """

    # Enforce the resolution requirements.
    if parsed_args.x_resolution == 1 and parsed_args.y_resolution == 1:
        logger.warn(
            "%s1x1 resolution is depreciated, consider running at a higher resolution."
            "%s",
            BColours.FAIL,
            BColours.ENDC,
        )
    elif (
        parsed_args.x_resolution >= (2 * number_of_pipes + 3)
        # and parsed_args.x_resolution % 2 == 1
        and parsed_args.y_resolution >= 3
    ):
        logger.info(
            "Resolution of %s by %s is accpetable and is greater than %s by 3.",
            parsed_args.x_resolution,
            parsed_args.y_resolution,
            int((2 * number_of_pipes + 3)),
        )
    else:
        logger.error(
            "%sThe specified resolution of %s by %s is not supported. The resolution "
            "must be either 1 by 1 or greater than %s by 3.%s",
            BColours.FAIL,
            parsed_args.x_resolution,
            parsed_args.y_resolution,
            int((2 * number_of_pipes + 3)),
            BColours.ENDC,
        )
        raise ArgumentMismatchError(
            "The specified resolution of {} by {} is not supported. The ".format(
                parsed_args.x_resolution, parsed_args.y_resolution
            )
            + "resolution must be either 1 by 1 or greater than {} by 3".format(
                int((2 * number_of_pipes + 3))
            )
        )

    # Enforce the matching up of dynamic and steady-state arguments.
    # Enforce that either dynamic or steady-state is specified, but not both.
    if (parsed_args.dynamic and parsed_args.steady_state) or (
        not parsed_args.dynamic and not parsed_args.steady_state
    ):
        raise ArgumentMismatchError(
            "{}Either `--dynamnic` or `--steady-state` should be used, but ".format(
                BColours.FAIL
            )
            + "not both.{}".format(BColours.ENDC)
        )

    # Enforce that, if decoupled is specified, steady-state is specified.
    if parsed_args.decoupled and not parsed_args.steady_state:
        raise ArgumentMismatchError(
            "{}If `--decoupled` is specified, the system must be run in ".format(
                BColours.FAIL
            )
            + "steady-state mode.{}".format(BColours.ENDC)
        )

    # Enforce that, if decoupled is specified, solar irradiance is specified.
    if parsed_args.decoupled and not parsed_args.steady_state_data_file:
        raise ArgumentMismatchError(
            "{}If `--decoupled` is specified, the steady-state data file must ".format(
                BColours.FAIL
            )
            + "be specified with `--steady-state-data-file`.{}".format(BColours.ENDC)
        )

    # Enforce that, if decoupled is not specified, either days or months is specified.
    if not parsed_args.decoupled and not (
        parsed_args.months is not None or parsed_args.days is not None
    ):
        raise ArgumentMismatchError(
            "{}If running a coupled system, the number of days or months for ".format(
                BColours.FAIL
            )
            + "the run must be specified with `--days` or `--months`.{}".format(
                BColours.ENDC
            )
        )

    # Enforce that, if decoupled is not specified, start-time is specified.
    if not parsed_args.decoupled and parsed_args.start_time is None:
        raise ArgumentMismatchError(
            "{}If running a coupled system, the start time for the run must be ".format(
                BColours.FAIL
            )
            + "specified with `--start-time`.{}".format(BColours.ENDC)
        )

    # Enforce the layer names are of the correct type.
    if not all((entry in layer_map for entry in parsed_args.layers)):
        raise ArgumentMismatchError(
            "{}If using the --layers developer argument, only layers {} can be "
            "specified. Rogue layer name: '{}'.{}".format(
                BColours.FAIL,
                ", ".join(layer_map.keys()),
                ", ".join(
                    [entry for entry in parsed_args.layers if entry not in layer_map]
                ),
                BColours.ENDC,
            )
        )

    return {layer_map[entry] for entry in parsed_args.layers}


def parse_args(args) -> argparse.Namespace:
    """
    Parse command-line arguments.

    :param args:
        The command-line arguments to parse.

    :return:
        The parsed arguments in an :class:`argparse.Namespace`.

    """

    parser = argparse.ArgumentParser()
    developer_arguments = parser.add_argument_group("developer arguments")
    dynamic_arguements = parser.add_argument_group("dynamic arguments")
    required_named_arguments = parser.add_argument_group("required named arguments")
    steady_state_arguments = parser.add_argument_group("steady-state arguments")

    steady_state_arguments.add_argument(
        "--ambient-temperature",
        "-at",
        type=float,
        help="[decoupled] The ambient temperature surrounding the absorber, in "
        "degrees Celcius, needs to be specified if running a decoupled system.",
    )
    parser.add_argument(
        "--average-irradiance",
        "-ar",
        action="store_true",
        default=False,
        help="Whether to average the solar irradiance intensities on a monthly basis, "
        "or to use the data for each day individually.",
    )
    steady_state_arguments.add_argument(
        "--collector-input-temperature",
        "-ci",
        default=None,
        type=float,
        help="[decoupled] The input temperature in degrees Celcius of HTF to use when "
        "modelling a decoupled PVT absorber.",
    )
    parser.add_argument(
        "--cloud-efficacy-factor",
        "-c",
        type=float,
        help="The effect that the cloud cover has, rated between 0 (no effect) and 1.",
    )
    dynamic_arguements.add_argument(
        "--days",
        "-d",
        type=int,
        help="The number of days to run the simulation for. Overrides 'months'.",
    )
    parser.add_argument(
        "--decoupled",
        action="store_true",
        default=False,
        help="If specified, the model will be run with a decoupled PVT absorber.",
    )
    parser.add_argument(
        "--dynamic",
        default=False,
        action="store_true",
        help="If specified, Fourier number calculation will be skipped and a dynamic "
        "model will be used for the run.",
    )
    dynamic_arguements.add_argument(
        "--exchanger-data-file",
        "-e",
        help="The location of the Exchanger system YAML data file.",
    )
    dynamic_arguements.add_argument(
        "--initial-month",
        "-i",
        type=int,
        help="The first month for which the simulation will be run, expressed as an "
        "int. The default is 1, corresponding to January.",
    )
    parser.add_argument(
        "--initial-system-temperature-vector",
        nargs="+",
        help="If specified, this will override the internal initial system "
        "temperature vector.",
    )
    developer_arguments.add_argument(
        "--layers",
        default=layer_map.keys(),
        nargs="+",
        type=str,
        help="Used to specifiy layers present. Options are 'g', 'pv', 'a', 'p', and "
        "'f'.",
    )
    required_named_arguments.add_argument(
        "--location", "-l", help="The location for which to run the simulation."
    )
    dynamic_arguements.add_argument(
        "--months",
        "-m",
        help="The number of months for which to run the simulation. Default is 12.",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--number-of-people",
        "-n",
        type=int,
        help="The number of household members to consider in this model.",
    )
    required_named_arguments.add_argument(
        "--output",
        "-o",
        help="The output file to save data to. This should be of JSON format.",
    )
    required_named_arguments.add_argument(
        "--portion-covered",
        "-pc",
        type=float,
        help="The portion of the panel which is covered in PV, "
        "from 1 (all) to 0 (none).",
    )
    dynamic_arguements.add_argument(
        "--pump-data-file",
        "-pm",
        help="The location of the pump YAML data file for the PV-T system pump.",
    )
    required_named_arguments.add_argument(
        "--pvt-data-file", "-p", help="The location of the PV-T system YAML data file."
    )
    dynamic_arguements.add_argument(
        "--resolution",
        "-r",
        help="The resolution, in seconds, used to solve the panel temperatures.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--return-system-data",
        help=argparse.SUPPRESS,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--skip-2d-output",
        "-sk2d",
        action="store_true",
        default=False,
        help="If specified, the 2D output will not be saved, and only 1D info, and "
        "plots, will be saved and generated.",
    )
    steady_state_arguments.add_argument(
        "--solar-irradiance",
        "-sol",
        default=None,
        type=float,
        help="[decoupled] The solar irradiance in Watts per meter squared to use when "
        "running the system as a decoupled PVT absorber.",
    )
    dynamic_arguements.add_argument(
        "--start-time",
        "-st",
        type=int,
        default=0,
        help="The start time, in hours, at which to begin the simulation during the day",
    )
    parser.add_argument(
        "--steady-state",
        "-ss",
        action="store_true",
        default=False,
        help="If specified, Fourier-number calculation will be skipped and the model "
        "will be run without transient/dynamic terms.",
    )
    steady_state_arguments.add_argument(
        "--steady-state-data-file",
        "-ssdf",
        default=None,
        type=str,
        help="The path to the data file containing information specifying the steady-"
        "state runs that should be carried out.",
    )
    dynamic_arguements.add_argument(
        "--tank-data-file",
        "-t",
        help="The location of the Hot-Water Tank system YAML data file.",
    )
    parser.add_argument(
        "--use-pvgis",
        default=False,
        action="store_true",
        help="If specified, PVGIS data is used. Otherwise, the data extracted from "
        "Maria's paper is used.",
    )
    parser.add_argument(
        "--unglazed",
        "-u",
        help="If specified, the panel will be un-glazed, i.e., without a glass coating.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="If specified, verbose logging will be carried out.",
    )
    required_named_arguments.add_argument(
        "--x-resolution",
        "-x",
        help="The number of segments to include in the x direction (across the panel) "
        "for the simulation.",
        type=int,
    )
    required_named_arguments.add_argument(
        "--y-resolution",
        "-y",
        help="The number of segments to include in the y direction (along the length of "
        "the panel) for the simulation.",
        type=int,
    )

    return parser.parse_args(args)
