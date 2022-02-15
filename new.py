import os
import shutil

from tqdm import tqdm

files_to_copy = {filename for filename in os.listdir(os.path.join("system_data", "steady_state_data")) if "bahraich" in filename}

for filename in tqdm(files_to_copy, desc="files", unit="file", leave=True):
    shutil.copy2(
	    os.path.join(
			"system_data",
			"steady_state_data",
			filename
		),
		os.path.join(
			"Y:",
			"home",
			"PVTModel",
			"system_data",
			"steady_state_data",
			filename
		)
	)
	
print("DONE")
