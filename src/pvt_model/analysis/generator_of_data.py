import yaml
from tqdm import tqdm

ambient_temperatures = {2.5 * entry for entry in range(0, 41, 1)}
max_collector_input_temperature = 80
mass_flow_rates = {0.5 * entry for entry in range(0, 41, 1)}
irradiances = set(range(0, 501, 100))
irradiances.update(set(range(510, 1011, 10)))
wind_speeds = set(range(0, 21, 1))

base_name = "system_data/steady_state_data/autotherm_parameter_space_v2_batch_{counter}"

for i, G in enumerate(tqdm(irradiances, desc="irradiance", leave=False)):
    for j, T_amb in enumerate(tqdm(ambient_temperatures, desc="ambient temp", leave=False)):
        for k, m_dot in enumerate(tqdm(mass_flow_rates, desc="mass flow rate", leave=False)):
            for l, v_w in enumerate(tqdm(wind_speeds, desc="wind speeds", leave=False)):
                runs = [
                    {
                        "ambient_temperature": T_amb,
                        "collector_input_temperature": T_c_in,
                        "mass_flow_rate": m_dot,
                        "solar_irradiance": G,
                        "wind_speed": v_w
                    }
                    for T_c_in in range(T_amb, max_collector_input_temperature, 4)
                ]
                with open(base_name.format(counter=str(max_collector_input_temperature / 4 + (len(wind_speeds) * (l + len(mass_flow_rates) * (k * len(ambient_temperatures) * (j + len(irradiances) * i)))))))
