#!/usr/bin/python3.7
########################################################################################
# hpc_pvt_model.py - Module for executing runs on the HPC.
#
# Author: Ben Winchester
# Copyright: Ben Winchester, 2021
########################################################################################

"""
Module for executing runs on the HPC.

Imperial College London provides high-performance computers for researchers to use. In
order to run the model on the HPC computers, the model must be called in such a way as
to wrap the model within a bash wrapper script.

This module takes care of all HPC-related wrapping that is required.

"""

import os
import subprocess
import sys
import tempfile


# The command prefix to use when calling qsub.
QSUB_COMMAND = "qsub"
# The name of the scripts directory.
SCRIPTS_DIRECTORY = "scripts"
# The name of the template HPC script.
TEMPLATE_HPC_SCRIPT_NAME = "template_hpc_pvt_model.sh"


def main() -> None:
    """
    The main method which contains all necessary wrapping code.

    """

    # Open the template HPC script.
    with open(os.path.join(SCRIPTS_DIRECTORY, TEMPLATE_HPC_SCRIPT_NAME), "r") as f:
        template_hpc_script_contents = f.read()

    # Substitute in the command-line arguments.
    substituted_hpc_script = template_hpc_script_contents.replace(
        "$@", " ".join(sys.argv[1:])
    )

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:

        # Save the temporary script.
        with tempfile.NamedTemporaryFile(
            mode="w+t", suffix=".sh", delete=False, dir=temp_dir
        ) as temp_file:
            temp_file.write(substituted_hpc_script)

        # Adjust the permissions and execute.
        os.chmod(temp_file.name, 0o775)
        subprocess.run(f"qsub {temp_file.name}", check=True)

        # Close the temporary file - this is not really necessary but saves on space.
        temp_file.close()


if __name__ == "__main__":
    main()
