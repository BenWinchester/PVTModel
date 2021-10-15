#!/bin/bash
#PBS -lwalltime=06:00:00
#PBS -lselect=1:ncpus=8:mem=2000Mb

echo -e "HPC script executed"

# Load the anaconda environment
module load anaconda3/personal
source activate py37

WORKING_DIR="$PBS_O_WORKDIR/output_files/large_parameter_space_probe"
CURRENT_DIR=$(pwd)
COMBINED_FILE="combined_data.json"
cd $WORKING_DIR

# Deleting existing combined file.
if [ -f $COMBINED_FILE ]
then
    echo -e "Combined file already exists, deleting..."
    rm $COMBINED_FILE
else
    echo -e "Creating does not already exist.y"
fi

echo -e "Running file combination script.@"
python foo.py

cd $PBS_O_WORKDIR

cp "$WORKING_DIR/$COMBINED_FILE" ../scp_outbound_dir

exit 0
