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
echo -e "Carrying out unglazed Ilaria analysis based on thesis."
python3.7 -m pvt_model --initial-month 9 --location system_data/london_ilaria/ \
    --pvt-data-file system_data/pvt_panels/ilarias_panel.yaml \
    --output output_files/june_week_2/ilarias_theroretical_panel_single_glazed \
    --x-resolution 31 --y-resolution 11 --decoupled --steady-state \
    --steady-state-data-file \
    system_data/steady_state_data/autotherm.yaml \
    --layers g pv a p f --portion-covered 1 --skip-analysis \
    --wind-speed 15

cd $CURRENT_DIR

exit 0
