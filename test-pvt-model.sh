#!/usr/bin/env bash
########################################################################################
# test-pvt-model.sh - Runs a series of tests across the system.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

echo "Running test suite: black, mypy, pylint, pytest, and enforcement."
echo -e "\e[1mRunning black...\e[0m"
black pvt_model/pvt_system_model
echo -e "\e[1mRunning mypy...\e[0m"
mypy pvt_model/pvt_system_model
echo -e "\e[1mRunning pylint on model code...\e[0m"
pylint pvt_model/pvt_system_model
echo -e "\e[1mRunning pylint on analysis code...\e[0m"
pylint pvt_model/analysis
echo -e "\e[1mRunning yamllint...\e[0m"
yamllint -c .yamllint-config.yaml system_data/
echo -e "\e[1mRunning pytest...\e[0m"
pytest pvt_model/pvt_system_model
echo -e "\e[1mRunning enforcement scripts...\e[0m"
/usr/bin/python3.7 pvt_model/enforcement/mypy_enforcement.py
echo -e "\e[1mTest suite complete: see above stdout for details.\e[0m"
