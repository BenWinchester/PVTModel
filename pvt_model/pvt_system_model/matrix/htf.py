#!/usr/bin/python3.7
########################################################################################
# htf.py - The htf module for the matrix component.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
The htf module for the matrix component.

This module computes and returns the equation(s) associated with the htf layer of the
PV-T collector for the matrix component.

The model works by arranging the system of differential equations as a matrix equation
such that the temperatures at each time step can be computed based on the coefficients
of the matrix which are computed based on the temperatures of the various components at
the previous time step, as well as various Physical and fundamental constants.

"""
