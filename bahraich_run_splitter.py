import os

import re
import yaml

from tqdm import tqdm

regex = re.compile(r"bahraich_run_\d")

filenames = os.listdir(
	os.path.join(
		"system_data",
		"steady_state_data",
	)
)

files_to_split = {filename for filename in filenames if regex.match(filename) is not None}

print(f"{len(files_to_split)} files to split")

for filename in tqdm(files_to_split, desc="files", unit="file", leave=True):
	with open(
        os.path.join(
            "system_data",
            "steady_state_data",
            filename
        ),
    "r") as f:
		data = yaml.safe_load(f)

	fileprefix = filename.split(".yaml")[0]

	for index in tqdm(range(16), desc="chunking", unit="chunk", leave=False):
		chunked_data = data[index::16]
		with open(
            os.path.join(
                "system_data",
                "steady_state_data",
                f"{fileprefix}_{index}.yaml"
            ),
            "w"
        ) as f:
			yaml.dump(chunked_data, f)

