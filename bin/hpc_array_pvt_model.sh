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

cat autotherm_10_pipe_runs.txt | head -n $PBS_ARRAY_INDEX | tail -n 1 | bash

