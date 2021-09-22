import json
import os
import re
import yaml

# Num increases:
#   The number of times everything needs to be scaled up.
NUM_INCREASES: int = 4

# Offset:
#   The amount to offset each run saved number by.
OFFSET: int = 176

# Steady-state directory:
#   The path to the steady-state directory.
STEADY_STATE_DIRECTORY: str = os.path.join("system_data", "steady_state_data")

# Run_regex:
#   Regex used to match the files.
run_regex = re.compile(r"redo_reduced_model_batch_(?P<run_number>\d*)\.yaml")

# Cycle through the relevant YAML files:
for filename in os.listdir(STEADY_STATE_DIRECTORY):
    print(f"File {filename} ", end="")
    match = run_regex.match(filename)
    if match is None:
        print("skipped...")
        continue

    # Parse out the run_number
    run_number = match.group("run_number")

    # Parse the filedata.
    with open(os.path.join(STEADY_STATE_DIRECTORY, filename), "r") as f:
        filedata = yaml.safe_load(f)
    print("parsed, ", end="")

    for increase in range(1, NUM_INCREASES + 1, 1):
        for entry in filedata:
            entry["collector_input_temperature"] += 20
        print("increased ", end="")
        # Save the output
        updated_filename = re.sub(
            str(run_number),
            str(int(run_number) + int(OFFSET * increase)),
            filename
        )
        with open(os.path.join(STEADY_STATE_DIRECTORY, updated_filename), "w") as f:
            yaml.dump(filedata, f)

        print("and saved, ", end="")
    print("done!")