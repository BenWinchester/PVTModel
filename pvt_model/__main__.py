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

from . import argparser
from .pvt_system_model.__main__ import main as pvt_system_model_main

from .__utils__ import get_logger, LOGGER_NAME


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

    # Determine the type of run from the arguments passed in.
    parsed_args, unknown_args = argparser.parse_args(args)

    # Determine the initial conditions for the run via iteration until the initial and
    # final temperatures for the day match up.
    system_data = pvt_system_model_main(unknown_args.append("--return-system-data"))


if __name__ == "__main__":
    main(sys.argv[1:])
