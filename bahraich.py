import os
import yaml

from tqdm import tqdm
from typing import Dict, List, Union

# Ambient temperature keyword
AMBIENT_TEMPERATURE: str = "ambient_temperature"

# Number of chunks to consider.
CHUNKS: int = 438

# Keyword for collector input temperature
COLLECTOR_INPUT_TEMPERATURE: str = "collector_input_temperature"

# Keyword for mass flow rate.
MASS_FLOW_RATE: str = "mass_flow_rate"

# Keyword for temperature
TEMPERATURE: str = "temperature"

with open(os.path.join("system_data", "steady_state_data", "bahraich_runs.yaml"), "r") as f:
   data = yaml.safe_load(f)

mass_flow_rates = list(range(0, 11, 1))

for chunk_index in tqdm(range(CHUNKS), desc="file chunk", unit="chunk"):
    data_chunk = data[chunk_index::CHUNKS]
    for m_rate in tqdm(mass_flow_rates, desc="flow rate", unit="flow rate", leave=False):
        new_data: List[Dict[str, Union[int, float]]] = []
        for entry in tqdm(data_chunk, desc="data entries", unit="entry", leave=False):
            for index, T_in in enumerate(range(int(entry[TEMPERATURE]), 81, 5)):
                new_entry = entry.copy()
                new_entry[AMBIENT_TEMPERATURE] = new_entry.pop(TEMPERATURE)
                new_entry[MASS_FLOW_RATE] = m_rate
                new_entry[COLLECTOR_INPUT_TEMPERATURE] = T_in
                new_data.append(new_entry)
        with open(os.path.join("system_data", "steady_state_data", f"bahraich_run_{chunk_index}_{m_rate}_{T_in}.yaml"), "w") as f:
            yaml.dump(new_data, f)
