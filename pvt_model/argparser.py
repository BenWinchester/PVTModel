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
        "--dynamic",
        default=False,
        action="store_true",
        help="If specified, Fourier number calculation will be skipped and a dynamic "
        "model will be used for the run.",
    )
    parser.add_argument(
        "--quasi-steady",
        default=False,
        action="store_true",
        help="If specified, Fourier number calculation will be skipped and a quasi-"
        "steady model will be used for the run.",
    )

    return parser.parse_known_args(args)
