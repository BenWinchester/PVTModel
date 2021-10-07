#!/bin/bash
########################################################################################
# hpc_array_pvt_model.sh.py - Runs the PV-T model in an HPC array job.                 #
#                                                                                      #
# Author(s): Ben Winchester                                                            #
# Copyright: Ben Winchester, 2021                                                      #
# Date created: 28/09/2021                                                             #
# License: Open source                                                                 #
# Most recent update: 14/07/2021                                                       #
#                                                                                      #
# For more information, please email:                                                  #
#     benedict.winchester@gmail.com                                                    #
########################################################################################

# Depending on the environmental variable, run the appropriate HPC job.

module load anaconda3/personal
source activate py37

OUTPUT_DIR="$PBS_O_WORKDIR/output_files"
CURRENT_DIR=$(pwd)
cd $PBS_O_WORKDIR

INPUT_FILE=$(cat steady_state_data_files.txt | head -n $PBS_ARRAY_INDEX | tail -n 1)
INPUT_FILE="new_steady_state_runs/$INPUT_FILE"
OUTPUT_FILE="output_files/large_parameter_space_probe/output_file_$PBS_ARRAY_INDEX"

python3.7 -u -m src.pvt_model --skip-analysis \
    --output $OUTPUT_FILE \
    --steady-state-data-file $INPUT_FILE \
    --decoupled --steady-state --initial-month 7 --location system_data/london_ilaria/ \
    --portion-covered 1 \
    --pvt-data-file system_data/pvt_panels/autotherm.yaml \
    --x-resolution 71 --y-resolution 31 --average-irradiance --skip-2d-output \
    --layers g pv a p f
