#PBS -J 530001-540000
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=8:mem=11800Mb

echo -e "HPC script executed"

# Load the anaconda environment
module load anaconda3/personal
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

echo -e "Running PV-T model"
./bin/hpc_array_pvt_model.sh
echo -e "PV-T model run successfully"
