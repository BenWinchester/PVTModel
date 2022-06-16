# PVTModel
2D model for a PVT system.

## Model basics.
The model can be run as a python module from the command-line interface. The model exposes:
* an overall method, for running the entire model including analysis;
* an analysis module, which can be executed to run the figure generation only;
* and an enforcement module.

The model aims to simulate the running of an integrated PV-T, hot-water, and load system. The system modelled consists of a PV-T panel component, a hot-water tank, and a system for simulating the demands placed on the system by the end user.

## Running the Model
The model can be executed with python3.7 from a command line by calling:
`python3.7 -m pvt_model`.

The model can be run as a decoupled panel or as an integrated (coupled) system.

For running the model, the following CLI arguments are required as a minimum:
* `--initial-month <month_number>` - Specifies the month for which weather data should be used.
* `--location <location_folder>` - Specifies the location-related information. For an example location, see the `system_data/london` folder.
* `--portion-covered <0-1>` - Must be used to specify the portion of the panel which is covered with PV cells.
* `--pvt-data-file <pv_data_file.yaml>` - The PV-T data YAML file must be specified.
* `--output <extension-independent output file name>` - The name of the output file, to which data should be saved, independent of file extension, should be specified.

For running the model as a stand-alone (decoupled) panel, the following requirements are required as an addition to the minimum above:
* `--decoupled --steady-state` - Must be used to specify that the run is decoupled and steady-state.
* `--steady-state-data-file <steady-state system data file>` - Information about the runs that should be conducted needs to be specified.
* `--x-resolution <int>` - Specifies the number of elements to use in the x-direction.
* `--y-resolution <int>` - Specifies the number of elements to use in the y-direction.

For running the model as an integrated (coupled) panel, the following requirements are required as an addition to the minimum above:
* `--dynamic` - Must be used to specifiy that the run is coupled and dynamic.
* `--exchanger-data-file <exchanger_data_file.yaml>` - Must be used to specify the YAML data file for the heat exchanger.
* `--resolution <int>` - The temporal resolution to use for the model.
* `--tank-data-file <tank_data_file.yaml>` - Must be used to specify the YAML data file for the hot-water tank.

For ease of use, the following command-line arguments are recommended when conducting a dynamic and coupled run:
* `--average-irradiance` - Stipulates that an average irradiance profile for the month must be used.
* `--start-time <int>` - The start time, in hours from midnight, for which to run the simularion. `0` is the default;

For help with the arguments needed in order to run the model, use the inbuilt help display:
`python3.7 -m pvt_model --help`.

__NOTE__: If you receive a `KeyError: <int>` on the command-line, it is likely that the argument `--average-irradiance` must be used. This is because solar-irradiance profiles are missing for certain days, and the command must be used to average over those profiles which are specified.

## Running the Analysis Module
The analysis module can also be run from the command-line interface. This should be executed as a python module: `py -m pvt_model.analysis -df <output_file_path_with_extension>`.

## Creating a Pull Request
All pull requests need to be approved by a repository administrator and need to pass a series of automated tests.

To confirm that your code will pass, run the scritp `test-pvt-model.sh` from the root of the repository to ensure that your code confirms to the standards required of the repository (regarding formatting and type annotations etc.), that all automated tests are passing, and that all `type: ignore` and `pylint: disable` flags are dedclared.

## HPC Support
__NOTE__: Support is included to run the model on Imperial's high-performance computing (HPC) system. Scripts for deploying runs are located in the `scripts` directory. This directory can be safely ignored when deploying the model on a home-PC setup.

### Copyright
Copyright Benedict Winchester, 2021
