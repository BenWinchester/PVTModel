#!/usr/bin/python3.7
########################################################################################
# __main__.py - The main module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The main module for the PV-T model.

This module coordinates the time-stepping of the itterative module, calling out to the
various components where necessary, as well as reading in the various data files and
processing the command-line arguments that define the scope of the model.

"""

import logging
import sys

from . import exchanger, load, pvt, tank, weather


# * Arg-parsing method


# * Potentially a generator-wrap around the time module.


def main(args) -> None:
    """
    The main module for the code.

    :param args:
        The command-line arguments passed into the component.

    """

    # * Set up logging with a file handler etc.

    # * Parse the system arguments from the commandline.

    # * Set-up the weather and load modules with the weather and load probabilities.

    # * Initialise the PV-T class, tank, exchanger, etc..

    # * Begin the itterative for loop

    for day_number, date, month in time:
        print("foo")
        # * Generate weather and load conditions from the load and weather classes.
        # * Call the pvt module to generate the new temperatures at this time step.
        # * Propogate this information through to the heat exchanger and pass in the
        # * tank s.t. it updates the tank correctly as well.
        # * Re-itterate through with the collector inlet temperature calculated based on
        # * the output temperature from the heat exchanger.

    # * Potentially generate some plots, and at least save the data.


if __name__ == "__main__":
    main(sys.argv[1:])
