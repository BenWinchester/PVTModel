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

from .__utils__ import BColours

__all__ = ("check_args", "parse_args")


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
) -> None:
    """
    Enforces rules on the command-line arguments passed in in addition to argparse rules

    :param parsed_args:
        The parsed command-line arguments.

    :param logger:
        The logger used for the run.

    :param number_of_pipes:
        The number of pipes attached to the collector being modelled.

    :raises: ArgumentMismatchError
        Raised if the command-line arguments mismatch.

    """

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
            "The specified resolution of {} by {} is not supported. The resolution ".format(
                parsed_args.x_resolution, parsed_args.y_resolution
            )
            + "must be either 1 by 1 or greater than {} by 3".format(
                int((2 * number_of_pipes + 3))
            )
        )


def parse_args(args) -> argparse.Namespace:
    """
    Parse command-line arguments.

    :param args:
        The command-line arguments to parse.

    :return:
        The parsed arguments in an :class:`argparse.Namespace`.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--average-irradiance",
        "-ar",
        action="store_true",
        default=False,
        help="Whether to average the solar irradiance intensities on a monthly basis, "
        "or to use the data for each day individually.",
    )
    parser.add_argument(
        "--cloud-efficacy-factor",
        "-c",
        type=float,
        help="The effect that the cloud cover has, rated between 0 (no effect) and 1.",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        help="The number of days to run the simulation for. Overrides 'months'.",
    )
    parser.add_argument(
        "--dynamic",
        default=False,
        action="store_true",
        help="If specified, Fourier number calculation will be skipped and a dynamic "
        "model will be used for the run.",
    )
    parser.add_argument(
        "--exchanger-data-file",
        "-e",
        help="The location of the Exchanger system YAML data file.",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--input-water-temperature",
        "-it",
        help="The input water temperature to instantiate the system, measured in "
        "Celcius. Defaults to 20 Celcius if not provided.",
    )
    parser.add_argument(
        "--location", "-l", help="The location for which to run the simulation."
    )
    parser.add_argument(
        "--months",
        "-m",
        help="The number of months for which to run the simulation. Default is 12.",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--no-pv",
        action="store_true",
        default=False,
        help="Used to specify a PV-T panel with no PV layer: ie, a Thermal collector.",
    )
    parser.add_argument(
        "--number-of-people",
        "-n",
        type=int,
        help="The number of household members to consider in this model.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="The output file to save data to. This should be of JSON format.",
    )
    parser.add_argument(
        "--portion-covered",
        "-pc",
        type=float,
        help="The portion of the panel which is covered in PV, "
        "from 1 (all) to 0 (none).",
    )
    parser.add_argument(
        "--pump-data-file",
        "-pm",
        help="The location of the pump YAML data file for the PV-T system pump.",
    )
    parser.add_argument(
        "--pvt-data-file", "-p", help="The location of the PV-T system YAML data file."
    )
    parser.add_argument(
        "--quasi-steady",
        default=False,
        action="store_true",
        help="If specified, Fourier number calculation will be skipped and a quasi-"
        "steady model will be used for the run.",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--start-time",
        "-st",
        type=int,
        default=0,
        help="The start time, in hours, at which to begin the simulation during the day",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--x-resolution",
        "-x",
        help="The number of segments to include in the x direction (across the panel) "
        "for the simulation.",
        type=int,
    )
    parser.add_argument(
        "--y-resolution",
        "-y",
        help="The number of segments to include in the y direction (along the length of "
        "the panel) for the simulation.",
        type=int,
    )

    return parser.parse_args(args)
