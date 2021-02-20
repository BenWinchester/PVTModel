# FirstPVTModel
First attempt at creating a model for a PVT system.

## Model basics.
The model can be run as a python module from the command-line interface. An analysis module, for creating figures for interpreting the data files produced by the model.

The model aims to simulate the running of an integrated PV-T, hot-water, and load system. The system modelled consists of a PV-T panel component, a hot-water tank, and a system for simulating the demands placed on the system by the end user.

## Running the Model
The model can be executed with python3.7 from a command line by calling:
`python3.7 -m pvt_model`.

At a minimum, the following arguments are needed:
* `--cloud-efficacy-factor <float>` - The effect to which the cloud cover should influence the solar irradiance;
* `--days <int>` - The number of days for which to run the model;
* `--exchanger-data-file <heat_exchangers/my_first_exchanger.yaml>` - Path to the heat-exchanger data file to use. An example data file is provided in the `heat_exchangers` folder.
* `--initial-month <int>` - The month for which to run the model, where 1 represents January and 12 December;
* `--location <location_folder_name>` - Path to the location folder to use. An example location, `london`, is provided;
* `--portion-covered <float>` - Portion of the panel that is covered with a photovoltaic layer;
* `--pvt-data-file <pvt_panels/marias_panel.yaml>` - Path to the pv data file;
* `--tank-data-file <tanks/my_first_hot_water_tank.yaml>` - Path to the tank data file. An example data file is provided in the `tanks` folder;
* `--resolution 1800` - The resolution, in seconds, for which to run the model. E.G., for an internal resolution of 5 minutes, `--resolution 300` should be used;
* `--output <output_file_path>` - The name and location, irrespective of file extension, for the output data from the model to be saved.

For ease of use, the following command-line arguments are recommended:
* `--average-irradiance` - Stipulates that an average irradiance profile for the month must be used.
* `--start-time <int>` - The start time, in hours from midnight, for which to run the simularion. `0` is the default;

For help with the arguments needed in order to run the model, use the inbuilt help display:
`python3.7 -m pvt_model --help`.

## Running the Analysis Module
The analysis module can also be run from the command-line interface. This should be executed as a python module: `py -m pvt_model.analysis -df <output_file_path_with_extension>`.

## Creating a Pull Request
All pull requests need to be approved by a repository administrator and need to pass a series of automated tests.

To confirm that your code will pass, run the scritp `test-pvt-model.sh` from the root of the repository to ensure that your code confirms to the standards required of the repository (regarding formatting and type annotations etc.), that all automated tests are passing, and that all `type: ignore` and `pylint: disable` flags are dedclared.

### Copyright
Copyright Benedict Winchester, 2021
