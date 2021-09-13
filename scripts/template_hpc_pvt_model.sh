#!/bin/bash
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=8:mem=11800Mb

echo -e "HPC script executed"

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
echo -e "Carrying out analysis with arguments: $@"
python3.7 -m pvt_model --skip-analysis $@

cd $CURRENT_DIR

exit 0
