#!/usr/bin/python3.7
########################################################################################
# __utils__.py - The utility module for this, my first, PV-T model! :O
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2020
########################################################################################

"""
The utility module for the PV-T model.

This module contains common functionality, strucutres, and types, to be used by the
various modules throughout the PVT model.

"""

import logging
import os

__all__ = (
    "get_logger",
    "LOGGER_NAME",
)

# The directory for storing the logs.
LOGGER_DIRECTORY = "logs`"
# The name used for the internal logger.
LOGGER_NAME = "pvt_model"


def get_logger(logger_name: str) -> logging.Logger:
    """
    Set-up and return a logger.

    :param logger_name:
        The name of the logger to instantiate.

    :return:
        The logger for the component.

    """

    # Create a logger with the current component name.
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Create the logging directory if it doesn't exist.
    if not os.path.isdir(LOGGER_DIRECTORY):
        os.mkdir(LOGGER_DIRECTORY)
    # Create a file handler which logs even debug messages.
    if os.path.exists(os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log")):
        os.rename(
            os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log"),
            os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log.1"),
        )
    fh = logging.FileHandler(os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log"))
    fh.setLevel(logging.DEBUG)
    # Create a console handler with a higher log level.
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # Create a formatter and add it to the handlers.
    formatter = logging.Formatter(
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
