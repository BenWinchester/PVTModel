#!/bin/bash
#PBS -lwalltime=12:00:00
#PBS -lselect=1:ncpus=8:mem=11800Mb

# Load the anaconda environment
module load anaconda3/py37
source activate py37

OUTPUT_DIR="$PBS_O_WORKDIR/output_files"
CURRENT_DIR=$(pwd)
cd $PBS_O_WORKDIR

# Create an output directory if it doesn't exist.
if [ -d $OUTPUT_DIR ]
then
    echo -e "Output directory already exists, this will be used."
else
    echo -e "Creating output directory"
    mkdir $OUTPUT_DIR
fi

# Sending more runs to the HPC
echo -e "Sending 'more runs' command."

py -m pvt_model --initial-month 7 --location system_data/london_ilaria --portion-covered 1 --pvt-data-file system_data/pvt_panels/ilarias_panel.yaml --output output_files/hpc_run_outputs/july_test_run_coupled_dynamic_ilaria_1_31_x_11 --x-resolution 31 --y-resolution 11 --dynamic --layers g pv a p f --days 1 --resolution 1800 --average-irradiance --start-time 0 --tank-data-file system_data/tanks/ilarias_hot_water_tank.yaml --exchanger-data-file system_data/heat_exchangers/ilarias_exchanger.yaml

cd $CURRENT_DIR
exit 0
