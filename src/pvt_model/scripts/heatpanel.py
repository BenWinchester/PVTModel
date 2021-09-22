#!/usr/bin/env python
#!/usr/bin/python3
########################################################################################
# heatpanel.py - Primary entry point for the heat-panel package.                       #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2021                                                      #
# Date created: 21/09/2021                                                             #
# License: Open source                                                                 #
########################################################################################
"""
heatpanel.py - Primary entry point for the heat-panel package.

"""

import sys

from ..__main__ import main as heat_panel


def main():
    """
    Main function of the CLOVER entry-point script.

    """

    heat_panel(sys.argv[1:])
