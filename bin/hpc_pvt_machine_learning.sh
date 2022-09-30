#!/bin/bash
########################################################################################
# hpc_pvt_machine_learning.sh.py - Runs the machine-learning fitting on the HPC.       #
#                                                                                      #
# Author(s): Ben Winchester                                                            #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 30/09/2022                                                             #
# License: Open source                                                                 #
# Most recent update: 30/09/2022                                                       #
#                                                                                      #
# For more information, please email:                                                  #
#     benedict.winchester@gmail.com                                                    #
########################################################################################

#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=8:mem=11800Mb

echo -e "HPC script executed."
# Depending on the environmental variable, run the appropriate HPC job.

module load anaconda3/personal
source activate py37

OUTPUT_DIR="$PBS_O_WORKDIR"
CURRENT_DIR=$(pwd)
cd $PBS_O_WORKDIR

INPUT_FILE="combined_data.json"
OUTPUT_DIR="machine_learning_outputs"

echo -e "Running machine-learning python program."

python3.7 -u -m src.pvt_model.analysis.machine_learning_fit -cv \
    -df $INPUT_FILE -f -np 14

cd $CURRENT_DIR

echo -e "Script exectued."

exit 0
