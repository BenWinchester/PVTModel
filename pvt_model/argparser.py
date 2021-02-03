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
        "--input-water-temperature",
        "-it",
        help="The input water temperature to instantiate the system, measured in "
        "Celcius. Defaults to 20 Celcius if not provided.",
    )
    parser.add_argument(
        "--internal-resolution",
        "-ir",
        help="The internal resolution, in seconds, used by the model.",
        type=int,
        default=1,
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

    return parser.parse_args(args)
