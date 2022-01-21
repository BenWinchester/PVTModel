"""
Module designed for temporary plot creation.

"""

import pickle
import json
import yaml

import numpy as np

from matplotlib import pyplot as plt        


with open("../CLOVER-2/src/clover/src/best_thermal_tree.sav", "rb") as f: 
    best_thermal_tree = pickle.load(f)

with open("../CLOVER-2/src/clover/src/best_electric_tree.sav", "rb") as f: 
    best_electric_tree = pickle.load(f) 

with open("autotherm_runs_manufacturer_validation_attempt_9.json", "r") as f:
    data = json.load(f)

data.pop("data_type")
runs = list(data.values())

with open("system_data/steady_state_data/autotherm_fast.yaml", "r") as f:
    data = yaml.safe_load(f)

###################
# Fitted equation #
###################

a_0 = 29.741604978319252
a_1 = -18.591187817656692
a_2 = 2.7665886796936974
a_3 = 21.26326604925854
a_4 = 2.369750239298799
a_5 = -5.682352509862814
a_6 = -0.44273532578369784
a_7 = 0.2028008441771341
a_8 = -0.011119875548502111
a_9 = 0.055071111041432544
a_10 = 0.08158562860833476
a_11 = -0.06020468104160886
a_12 = -5.781350419589919
a_13 = 2.1957501679949383
a_14 = -0.29053071799402297
a_15 = -1.8642000928076754
a_16 = -0.2168704193965011
a_17 = 0.6040946267503874
a_18 = 0.3503231839458856
a_19 = -0.09602338634428392
a_20 = 0.007427401892109994
a_21 = -0.0376976418177416
a_22 = -0.0069869817616982095
a_23 = 0.005502093603775405
a_24 = 0.5260015449448561
a_25 = 0.008404945236477125
a_26 = -0.0009232365144613755
a_27 = 0.29551049432336596
a_28 = -0.06977464822399262
a_29 = 0.0054089335282716255

b_0 = 0.11639738048249704
b_1 = 0.013316120967158817
b_2 = -0.0017275968821738896
b_3 = -0.007118345534937452
b_4 = -0.0009603753775774027
b_5 = 0.001963935108755282
b_6 = 0.00013088241772800534
b_7 = -8.21290503977136e-05
b_8 = 4.953260935491807e-06
b_9 = -2.4716212339190652e-05
b_10 = -3.254925000717009e-05
b_11 = 2.2966026636676483e-05
b_12 = 0.0039941028007474885
b_13 = -0.0013675551372885565
b_14 = 0.00016088723949178542
b_15 = 0.0005934066665735958
b_16 = 0.00011959274164597139
b_17 = -0.00021214131631117
b_18 = -0.00018182934054386044
b_19 = 5.080180915525177e-05
b_20 = -4.0790614401728095e-06
b_21 = 1.5313147039884785e-05
b_22 = 1.4287639500124582e-06
b_23 = -2.007075139853903e-06
b_24 = -0.0004418125413464358
b_25 = 3.6827017656719075e-06
b_26 = -6.512317845834291e-07
b_27 = -0.00011076377100134386
b_28 = 2.8315159160794437e-05
b_29 = -1.5880243199525098e-06

for entry in runs:
    entry["five_param_therm"] = (
        a_0
        + a_1 * np.log(entry["solar_irradiance"])
        + a_2 * (np.log(entry["solar_irradiance"])) ** 2
        + a_3 * np.log(3600 * entry["mass_flow_rate"])
        + a_4 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
        + a_5 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        + entry["ambient_temperature"] * (
            a_6
            + a_7 * np.log(entry["solar_irradiance"])
            + a_8 * (np.log(entry["solar_irradiance"])) ** 2
            + a_9 * np.log(3600 * entry["mass_flow_rate"])
            + a_10 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
            + a_11 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        )
        + entry["wind_speed"] ** 0.16 * (
            a_12
            + a_13 * np.log(entry["solar_irradiance"])
            + a_14 * (np.log(entry["solar_irradiance"])) ** 2
            + a_15 * np.log(3600 * entry["mass_flow_rate"])
            + a_16 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
            + a_17 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        )
        + entry["ambient_temperature"] * entry["wind_speed"] ** 0.16 * (
            a_18
            + a_19 * np.log(entry["solar_irradiance"])
            + a_20 * (np.log(entry["solar_irradiance"])) ** 2
            + a_21 * np.log(3600 * entry["mass_flow_rate"])
            + a_22 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
            + a_23 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        )
        + entry["collector_input_temperature"] * (
            a_24
            + a_25 * np.log(entry["solar_irradiance"])
            + a_26 * (np.log(entry["solar_irradiance"])) ** 2
            + a_27 * np.log(3600 * entry["mass_flow_rate"])
            + a_28 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
            + a_29 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        )
    )
    entry["five_param_electric"] = (
        b_0
        + b_1 * np.log(entry["solar_irradiance"])
        + b_2 * (np.log(entry["solar_irradiance"])) ** 2
        + b_3 * np.log(3600 * entry["mass_flow_rate"])
        + b_4 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
        + b_5 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        + entry["ambient_temperature"] * (
            b_6
            + b_7 * np.log(entry["solar_irradiance"])
            + b_8 * (np.log(entry["solar_irradiance"])) ** 2
            + b_9 * np.log(3600 * entry["mass_flow_rate"])
            + b_10 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
            + b_11 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        )
        + entry["wind_speed"] ** 0.16 * (
            b_12
            + b_13 * np.log(entry["solar_irradiance"])
            + b_14 * (np.log(entry["solar_irradiance"])) ** 2
            + b_15 * np.log(3600 * entry["mass_flow_rate"])
            + b_16 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
            + b_17 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        )
        + entry["ambient_temperature"] * entry["wind_speed"] ** 0.16 * (
            b_18
            + b_19 * np.log(entry["solar_irradiance"])
            + b_20 * (np.log(entry["solar_irradiance"])) ** 2
            + b_21 * np.log(3600 * entry["mass_flow_rate"])
            + b_22 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
            + b_23 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        )
        + entry["collector_input_temperature"] * (
            b_24
            + b_25 * np.log(entry["solar_irradiance"])
            + b_26 * (np.log(entry["solar_irradiance"])) ** 2
            + b_27 * np.log(3600 * entry["mass_flow_rate"])
            + b_28 * (np.log(3600 * entry["mass_flow_rate"])) ** 2
            + b_29 * np.log(entry["solar_irradiance"]) * np.log(3600 * entry["mass_flow_rate"])
        )
    )

##############################
# Standard performance model #
##############################


############
# AI model #
############

for entry in runs:
    entry["best_thermal_tree"] = float(best_thermal_tree.predict([[entry["ambient_temperature"], entry["collector_input_temperature"], 3600 * entry["mass_flow_rate"], entry["solar_irradiance"], entry["wind_speed"]]]))
    entry["best_electric_tree"] = float(best_electric_tree.predict([[entry["ambient_temperature"], entry["collector_input_temperature"], 3600 * entry["mass_flow_rate"], entry["solar_irradiance"], entry["wind_speed"]]]))

for entry in data:
    entry["mass_flow_rate"] = 14 * entry["mass_flow_rate"]

# Plotting the hopeful graph.
plt.scatter([entry["collector_input_temperature"] for entry in runs], [entry["best_thermal_tree"] for entry in runs], marker="+", label="reduced model", color="C2")
plt.scatter([entry["collector_input_temperature"] for entry in data], [entry["collector_input_temperature"] + entry["collector_temperature_gain"] for entry in data], marker="x", label="manufacturer data", color="C0")
plt.scatter([entry["collector_input_temperature"] for entry in runs], [entry["collector_output_temperature"] for entry in runs], marker="3", label="technical model", color="C1")
plt.scatter([entry["collector_input_temperature"] for entry in runs], [entry["five_param_therm"] for entry in runs], marker="1", label="five param fitting", color="C3")
plt.xlabel("Collector input temperature / degC")
plt.ylabel("Collector output temperature / degC")
plt.title("Comparison of technical and reduced models to manufacturer autotherm's data.")
plt.legend()
plt.show()

###############################################################
# Plot the output temperature against the reduced temperature #
###############################################################

for entry in runs:
    entry["reduced_temperature"] = (0.5 * (entry["collector_input_temperature"] + entry["collector_output_temperature"]) - entry["ambient_temperature"]) / entry["solar_irradiance"]
    entry["thermal_tree_reduced_temperature"] = (0.5 * (entry["collector_input_temperature"] + entry["best_thermal_tree"]) - entry["ambient_temperature"]) / entry["solar_irradiance"]
    entry["five_param_reduced_temperature"] = (0.5 * (entry["collector_input_temperature"] + entry["five_param_therm"]) - entry["ambient_temperature"]) / entry["solar_irradiance"]

for entry in data:
    entry["reduced_temperature"] = (0.5 * (2 * entry["collector_input_temperature"] + entry["collector_temperature_gain"]) - entry["ambient_temperature"]) / entry["irradiance"]

plt.scatter([entry["reduced_temperature"] for entry in runs], [entry["collector_output_temperature"] for entry in runs], marker="3", label="technical model", color="C1")
plt.scatter([entry["reduced_temperature"] for entry in data], [entry["collector_input_temperature"] + entry["collector_temperature_gain"] for entry in data], marker="x", label="manufacturer data", color="C0")
plt.scatter([entry["thermal_tree_reduced_temperature"] for entry in runs], [entry["best_thermal_tree"] for entry in runs], marker="+", label="reduced model", color="C2")
plt.scatter([entry["five_param_reduced_temperature"] for entry in runs], [entry["five_param_therm"] for entry in runs], marker="1", label="five param fitting", color="C3")
plt.legend()
plt.show()

###############################################################
# Plot the thermal efficiency against the reduced temperature #
###############################################################

for entry in data:
    entry["thermal_efficiency"] = (4180 * (entry["mass_flow_rate"] / (14 * 3600)) * ( entry["collector_temperature_gain"])) / (entry["irradiance"] * 0.1019714285572)

for entry in runs:
    entry["thermal_efficiency"] = (4180 * entry["mass_flow_rate"] * (entry["collector_output_temperature"] - entry["collector_input_temperature"])) / (entry["solar_irradiance"] * 0.1019714285572)
    entry["thermal_tree_thermal_efficiency"] = (4180 * entry["mass_flow_rate"] * (entry["best_thermal_tree"] - entry["collector_input_temperature"])) / (entry["solar_irradiance"] * 0.1019714285572)
    entry["five_param_thermal_efficiency"] = (4180 * entry["mass_flow_rate"] * (entry["five_param_therm"] - entry["collector_input_temperature"])) / (entry["solar_irradiance"] * 0.1019714285572)

plt.scatter([entry["reduced_temperature"] for entry in runs], [entry["thermal_efficiency"] for entry in runs], marker="3", label="technical model", color="C1")
plt.scatter([entry["reduced_temperature"] for entry in data], [entry["thermal_efficiency"] for entry in data], marker="x", label="manufacturer data", color="C0")
plt.scatter([entry["thermal_tree_reduced_temperature"] for entry in runs], [entry["thermal_tree_thermal_efficiency"] for entry in runs], marker="+", label="reduced model", color="C2")
plt.scatter([entry["five_param_reduced_temperature"] for entry in runs], [entry["five_param_thermal_efficiency"] for entry in runs], marker="1", label="five param fitting", color="C3")
plt.legend()
plt.show()
