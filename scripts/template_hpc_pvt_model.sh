#PBS -lwalltime=00:10:00
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

# Sending more runs to the HPC
echo -e "Carrying out analysis with arguments: "
python3.7 -u -m src.pvt_model --skip-analysis --output output_files/bahraich_parameter_probe/bahraich_output_file_test --steady-state-data-file system_data/steady_state_data/bahraich_run_213_9_78_1.yaml  --decoupled --steady-state --initial-month 7 --location system_data/london_ilaria/ --portion-covered 1 --pvt-data-file system_data/pvt_panels/autotherm.yaml --x-resolution 31 --y-resolution 11 --average-irradiance --skip-2d-output --layers g pv a p f


cd $CURRENT_DIR

exit 0
