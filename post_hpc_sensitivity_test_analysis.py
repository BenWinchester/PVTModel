#!/usr/bin/python3

import json
import math
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

from matplotlib import rc, rcParams
from pycirclize import Circos
from tqdm import tqdm

COMBINED_CSV_FILENAME: str = "combined_17_may_post_hpc_sensitivity_test.csv"
INDEX: int = 1
TOTEX_HEADER: str = "Other TOTEX"

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})
# rc("figure", **{"figsize": (48 / 5, 32 / 5)})
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
sns.set_context("notebook")
sns.set_style("whitegrid")

# Set custom color-blind colormap

colorblind_palette = sns.color_palette(
    [
        "#E04606",  # Orange
        "#F09F52",  # Pale orange
        "#52C0AD",  # Pale green
        "#006264",  # Green
        "#144E56",
        "#D8247C",  # Pink
        "#EDEDED",  # Pale pink
        "#E7DFBE",  # Pale yellow
        "#FBBB2C",  # Yellow
    ]
)

sns.set_palette(colorblind_palette)

colorblind_plus_palette = sns.color_palette(
    [
        "#E04606",  # Orange
        "#F09F52",  # Pale orange
        "#EFF2DD",
        "#52C0AD",  # Pale green
        "#006264",  # Green
        "#144E56",
        "#D8247C",  # Pink
        "#EDEDED",  # Pale pink
        "#E7DFBE",  # Pale yellow
        "#FBBB2C",  # Yellow
    ]
)

thesis_palette = sns.color_palette(
    [
        "#E04606",
        "#F9A130",
        "#EFF2DD",
        "#27BFE6",  # SDG 6
        "#006264",  # Green
        "#144E56",
        "#D8247C",  # Pink
        "#EDEDED",  # Pale pink
        "#E7DFBE",  # Pale yellow
        "#FBBB2C",  # Yellow
    ]
)

blue_first_thesis_palette = sns.color_palette(
    [
        "#006264",  # Green
        "#27BFE6",  # SDG 6
        "#EFF2DD",
        "#F9A130",  # Orange
        "#E04606",  # Dark orange
        # "#F5B112",  # SDG 7
        "#E7DFBE",  # Pale yellow
        "#EDEDED",  # Pale pink
        "#D8247C",  # Pink
        "#FBBB2C",  # Yellow
    ]
)

# Expand the thesis blue-first palette to have double the number of colours
blue_first_thesis_colours = blue_first_thesis_palette.as_hex()

colourmap = mcolors.LinearSegmentedColormap.from_list(
    "", np.array(blue_first_thesis_colours), 20
)

blue_first_thesis_palette_7 = sns.color_palette(
    colourmap(np.linspace(0, 1, 7)).tolist()
)
sns.set_palette(blue_first_thesis_palette_7)

blue_first_thesis_palette_18 = sns.color_palette(
    colourmap(np.linspace(0, 1, 18)).tolist()
)
sns.set_palette(blue_first_thesis_palette_18)

blue_first_thesis_palette_20 = sns.color_palette(
    colourmap(np.linspace(0, 1, 20)).tolist()
)
sns.set_palette(blue_first_thesis_palette_20)

# Figure size
# fig, axes = plt.subplots(2, 2, figsize=(48 / 5, 32 / 5))
fig = plt.figure(figsize=(48 / 5, 32 / 5))

##############################
# Fit of reduced temperature #
##############################

# For this fit of reduced temperature, a file can be opened from the analysis, and the
# thermal efficiency of the collector at various reduced-temperature values can be
# plotted. It's then possible to have the reduecd temperature quadratic curves added to
# the plot.

with open(
    (
        reduced_temperature_plot_filename := os.path.join(
            "output_files",
            "17_may_24",
            "autotherm_sensitivity_of_glass_thermal_conductivity_with_value_0.58.json",
        )
    )
) as f:
    frame_to_plot = pd.DataFrame(
        [value for key, value in json.load(f).items() if "run" in key]
    )

reduced_temperature_fit = np.poly1d(
    np.polyfit(
        frame_to_plot["reduced_collector_temperature"].dropna(),
        frame_to_plot["thermal_efficiency"].dropna(),
        2,
    )
)

frame_to_plot["marker"] = [
    "h" if row[1]["wind_speed"] == 1 else "H" if row[1]["wind_speed"] == 5 else "D"
    for row in frame_to_plot.iterrows()
]

fig = plt.figure(figsize=(48 / 5, 32 / 5))
for marker, sub_frame in frame_to_plot.groupby("marker"):
    plt.scatter(
        [],
        [],
        alpha=0,
        label=f"Wind speed = {'1' if marker == 'h' else '5' if marker == 'H' else '10'} m/s",
    )
    sns.scatterplot(
        sub_frame,
        x="reduced_collector_temperature",
        y="thermal_efficiency",
        hue="collector_input_temperature",
        marker=marker,
        alpha=0.5,
        s=200,
        palette=sns.color_palette(
            "RdBu_r" if marker == "h" else "PiYG_r" if marker == "H" else "PRGn", 4
        ),
        linewidth=0,
    )
    # sub_reduced_temperature_fit = np.poly1d(
    #     np.polyfit(
    #         sub_frame["reduced_collector_temperature"],
    #         sub_frame["thermal_efficiency"],
    #         2,
    #     )
    # )
    # sns.lineplot(
    #     x=(
    #         x_linspace := np.linspace(
    #             min(sub_frame["reduced_collector_temperature"]),
    #             max(sub_frame["reduced_collector_temperature"]),
    #         )
    #     ),
    #     y=sub_reduced_temperature_fit(x_linspace),
    #     color="C4",
    #     dashes=(2, 2),
    #     label="Quadratic fit",
    # )

plt.xlabel("Reduced collector temperature / Km$^2$/W")
plt.ylabel("Thermal efficiency")
plt.legend(
    title="HTF input temperature / $^\circ$C",
    fontsize="medium",
    title_fontsize="medium",
)

plt.savefig(
    "17_may_"
    + reduced_temperature_plot_filename.split("autotherm_sensitivity_of_")[1]
    + f"_{INDEX}.png",
    transparent=True,
    bbox_inches="tight",
    dpi=400,
)

plt.show()

#############################################
# Sensitivity analysis of parameters varied #
#############################################

# This space will be used to plot the sensitivity of the thermal efficiency to the
# various parameters that were explored.

filename_regex = re.compile(
    r"autotherm_sensitivity_of_(?P<component_name>absorber|adhesive|air_gap|bond|eva|glass|insulation|pipe|pv|pvt|tedlar)_(?P<parameter_name>.*)_with_value_(?P<value>.*)\.json"
)
run_number_regex = re.compile(r"run_(?P<run_number>\d*)_T.*")

# with open((benchmark_filename := "15_mar_benchmark.json")) as benchmark_file:
with open((benchmark_filename := "30_apr_benchmark.json")) as benchmark_file:
    benchmark_data = json.load(benchmark_file)

if not os.path.isfile(COMBINED_CSV_FILENAME):
    un_concatenated_dataframes: list[pd.DataFrame] = []
    for filename in tqdm(
        os.listdir(
            (output_directory_name := os.path.join("output_files", "17_may_24"))
        ),
        desc="processing files",
        unit="files",
        leave=True,
    ):
        with open(os.path.join(output_directory_name, filename)) as output_file:
            # Parse the file data
            loaded_data = json.load(output_file)
        file_dataframe = pd.DataFrame(
            [value for key, value in loaded_data.items() if "run" in key]
        )
        # Add the run number as a column
        try:
            run_number = [
                run_number_regex.match(entry).group("run_number")
                for entry in [
                    sub_entry for sub_entry in loaded_data.keys() if "run" in sub_entry
                ]
            ]
        except AttributeError:
            print("Couldn't match for run name.")
            raise
        file_dataframe["run_number"] = run_number
        # Determine the parameter being varied and the current value of that parameter
        match = filename_regex.match(filename)
        if match is None:
            raise Exception(
                f"Could not match the filename '{filename}' with the regex."
            )
        # Save the component name, parameter name, and value to the dataframe.
        file_dataframe["component_name"] = match.group("component_name")
        file_dataframe["parameter_name"] = match.group("parameter_name")
        file_dataframe["parameter_value"] = match.group("value")
        # Append this to the dict of data
        un_concatenated_dataframes.append(file_dataframe)
    combined_output_frame = pd.concat(un_concatenated_dataframes).reset_index(drop=True)
    del un_concatenated_dataframes
    # Produce a mapping between run number and the thermal efficiency for the benchmark
    # data.
    benchmark_thermal_efficiency_map = {
        run_number_regex.match(key).group("run_number"): value["thermal_efficiency"]
        for key, value in {
            k: v for k, v in benchmark_data.items() if "run" in k
        }.items()
    }
    # Determine the percentage difference (increase and decrease welcome) as well as the
    # variance in the value from the benchmark value for the thermal efficiency for each
    # entry in the dataframe.
    combined_output_frame["fractional_thermal_efficiency_change"] = None
    combined_output_frame["fractional_parameter_value_change"] = None
    combined_output_frame["percentage_thermal_efficiency_change"] = None
    combined_output_frame["squared_thermal_efficiency_change"] = None
    for index, row in tqdm(
        combined_output_frame.iterrows(),
        desc="reduced-temperature calculation",
        unit="row",
        leave=True,
        total=len(combined_output_frame),
    ):
        # Compute the values
        try:
            _fractional_thermal_efficiency_change = (
                row["thermal_efficiency"]
                - (
                    _this_benchmark_value := benchmark_thermal_efficiency_map[
                        row["run_number"]
                    ]
                )
            ) / _this_benchmark_value
        except KeyError:
            continue
        except TypeError:
            combined_output_frame.at[index, "fractional_thermal_efficiency_change"] = (
                None
            )
            combined_output_frame.at[index, "percentage_thermal_efficiency_change"] = (
                None
            )
            combined_output_frame.at[index, "squared_thermal_efficiency_change"] = None
            continue
        _percentage_thermal_efficiency_change = (
            100 * _fractional_thermal_efficiency_change
        )
        _squared_thermal_efficiency_change = _fractional_thermal_efficiency_change**2
        # Set the values
        combined_output_frame.at[index, "fractional_thermal_efficiency_change"] = (
            _fractional_thermal_efficiency_change
        )
        combined_output_frame.at[index, "percentage_thermal_efficiency_change"] = (
            _percentage_thermal_efficiency_change
        )
        combined_output_frame.at[index, "squared_thermal_efficiency_change"] = (
            _squared_thermal_efficiency_change
        )
    # Sanitise the parameter and componet names
    combined_output_frame["component_name"] = [
        entry.replace("_", " ").capitalize()
        for entry in combined_output_frame["component_name"]
    ]
    combined_output_frame["parameter_name"] = [
        entry.replace("_", " ").capitalize()
        for entry in combined_output_frame["parameter_name"]
    ]
    with open(COMBINED_CSV_FILENAME, "w") as combined_outputfile:
        combined_output_frame.to_csv(combined_outputfile)
else:
    with open(COMBINED_CSV_FILENAME, "r") as combined_outputfile:
        combined_output_frame = pd.read_csv(combined_outputfile, index_col=0)

def _temp(data):
    return [np.sqrt(entry) if entry is not None else None for entry in data]

combined_output_frame["abs_thermal_efficiency_change"] = _temp(combined_output_frame["squared_thermal_efficiency_change"])

###############################################
# Plot a marginal abatement curve--style plot #
###############################################

blue_first_thesis_palette_20 = sns.color_palette(
    colourmap(np.linspace(0, 1, 20)).tolist()
)
sns.set_palette(blue_first_thesis_palette_20)

def _fix_component_name(name_to_fix: str) -> str:
    """
    Fix un-capitalised names.

    Inputs:
        - name_to_fix:
            The name to fix.

    Returns:
        The fixed name.

    """
    match name_to_fix:
        case "Pv":
            return "PV"
        case "Pvt":
            return "PV-T"
        case "Eva":
            return "EVA"
        case _:
            return name_to_fix  \


combined_output_frame["component_name"] = list(map(_fix_component_name, combined_output_frame["component_name"]))

combined_output_frame["combined_name"] = (
    combined_output_frame["component_name"]
    + " "
    + combined_output_frame["parameter_name"]
)

sns.set_style("whitegrid")
sns.set_context("notebook")

# Exclude parameters that have no impact in order to reduce empty space
# Ugly I know!!
masked_combined_output_frame = combined_output_frame[combined_output_frame.combined_name != "Absorber Heat capacity"]
masked_combined_output_frame = masked_combined_output_frame[masked_combined_output_frame.combined_name != "Glass Density"]
masked_combined_output_frame = masked_combined_output_frame[masked_combined_output_frame.combined_name != "Glass Heat capacity"]
masked_combined_output_frame = masked_combined_output_frame[masked_combined_output_frame.combined_name != "PV Density"]
masked_combined_output_frame = masked_combined_output_frame[masked_combined_output_frame.combined_name != "PV Heat capacity"]

# Generate a combined plot with the marker size and shape indicating the irradiance.
parameter_median_squared_fractional_changes: dict[tuple[str, str], float] = {}
for component_name, sub_frame in tqdm(
    masked_combined_output_frame[masked_combined_output_frame["solar_irradiance"] == 1000].groupby("component_name"), leave=True
):
    for parameter_name, sub_sub_frame in tqdm(
        sub_frame.groupby("parameter_name"), leave=False
    ):
        parameter_median_squared_fractional_changes[
            component_name, parameter_name
        ] = float(sub_sub_frame["abs_thermal_efficiency_change"].median())

sorted_medians = sorted(
    parameter_median_squared_fractional_changes.items(), key=lambda item: item[1]
)

median_values_by_irradiance = {G: {combined_name: sub_frame["abs_thermal_efficiency_change"].median() for combined_name, sub_frame in masked_combined_output_frame[masked_combined_output_frame["solar_irradiance"] == G].groupby("combined_name")} for G in [200, 400, 600, 800, 1000]}
median_frame_to_plot = pd.concat([pd.DataFrame({"median_abs_change": median_values_by_irradiance[G], "Solar irradiance / W/m$^2$": G}) for G in [200, 400, 600, 800, 1000]], axis=0)
median_frame_to_plot["Component"] = [entry.split(" ")[0].replace("Air", "Air gap") for entry in list(median_frame_to_plot.index)]

fig = plt.figure(figsize=(32 / 5, 32 / 5))
ax = plt.gca()
sns.scatterplot(
    (frame_to_plot:=median_frame_to_plot.sort_values(by="median_abs_change", ascending=False)),
    y=frame_to_plot.index,
    x="median_abs_change",
    hue="Component",
    hue_order=sorted(set(combined_output_frame["component_name"])),
    markers=["h", "H", "s", "D", "P"],
    s=150,
    alpha=0.5,
    style="Solar irradiance / W/m$^2$",
    ax=ax,
    edgecolor="black"
)

# ax.tick_params(axis="x", rotation=90)
ax.set_ylabel("")
ax.set_xlabel("Median absolute fractional change in thermal efficiency")
ax.set_xscale("log")
ax.axvline(0.01, linestyle="--", color="grey", label="1% impact")
ax.axvline(0.042, linestyle="-.", color="grey", label="4.2% impact")
ax.set_xlim((min_ylim:=10**(-4)), 1/3)
ax.axvspan(min_ylim, 0.042,     color="grey",
zorder=0,
label="Rejected parameters", alpha=0.1)
ax.axvspan(0.01, 0.042,     color="C0",
zorder=0,
label="Limited impact", alpha=0.1)
legend = ax.legend(fancybox=True, framealpha=1, facecolor="white", bbox_to_anchor=(1, 1))
# legend.set_title("Component")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_alpha(1)
plt.savefig(
    f"abs_scatter_sensitivity_autotherm_all_irradiances_{INDEX}.png",
    transparent=True,
    format="png",
    dpi=400,
    bbox_inches="tight",
)


# fig = plt.figure(figsize=(48 / 5, 32 / 5))
# ax = plt.gca()
# sns.scatterplot(
#     (frame_to_plot:=median_frame_to_plot.sort_values(by="median_abs_change", ascending=True)),
#     x=frame_to_plot.index,
#     y="median_abs_change",
#     hue="Component",
#     hue_order=sorted(set(combined_output_frame["component_name"])),
#     marker="h",
#     alpha=0.5,
#     s=200,
#     size="Solar irradiance / W/m$^2$",
#     ax=ax,
# )

# ax.tick_params(axis="x", rotation=90)
# ax.set_xlabel("")
# ax.set_ylabel("Median absolute fractional change in thermal efficiency")
# ax.set_yscale("log")
# ax.axhline(0.01, linestyle="--", color="grey", label="1% impact")
# ax.axhline(0.042, linestyle="-.", color="grey", label="4.2% impact")
# ax.set_ylim((min_ylim:=10**(-13)), 10)
# ax.axhspan(min_ylim, 0.042,     color="grey",
# zorder=0,
# label="Rejected parameters", alpha=0.1)
# ax.axhspan(0.01, 0.042,     color="C0",
# zorder=0,
# label="Limited impact", alpha=0.1)
# legend = ax.legend(loc='lower right', fancybox=True, framealpha=1, facecolor="white")
# # legend.set_title("Component")
# frame = legend.get_frame()
# frame.set_facecolor("white")
# frame.set_alpha(1)
# plt.savefig(
#     f"abs_scatter_sensitivity_autotherm_all_irradiances_{INDEX}.png",
#     transparent=True,
#     format="png",
#     dpi=300,
#     bbox_inches="tight",
# )

median_frame_to_plot = median_frame_to_plot.sort_values(by="median_abs_change", ascending=True)


# Generate separate plots disaggregated by irradiance
for G in tqdm([200, 400, 600, 800, 1000]):
    parameter_median_squared_fractional_changes: dict[tuple[str, str], float] = {}
    parameter_mean_squared_fractional_changes: dict[tuple[str, str], float] = {}
    frame_to_plot = masked_combined_output_frame[
        masked_combined_output_frame["solar_irradiance"] == G
    ]
    for component_name, sub_frame in tqdm(
        frame_to_plot.groupby("component_name"), leave=True
    ):
        for parameter_name, sub_sub_frame in tqdm(
            sub_frame.groupby("parameter_name"), leave=False
        ):
            parameter_median_squared_fractional_changes[
                component_name, parameter_name
            ] = float(sub_sub_frame["abs_thermal_efficiency_change"].median())
            parameter_mean_squared_fractional_changes[
                component_name, parameter_name
            ] = float(sub_sub_frame["abs_thermal_efficiency_change"].median())
    sorted_medians = sorted(
        parameter_median_squared_fractional_changes.items(), key=lambda item: item[1]
    )
    sorted_means = sorted(
        parameter_mean_squared_fractional_changes.items(), key=lambda item: item[1]
    )
    # g = sns.catplot(x=["_".join(entry[0]) for entry in sorted_medians], y=[entry[1] for entry in sorted_medians], hue=[entry[0][0] for entry in sorted_medians], kind="bar")
    gng = sns.catplot(
        frame_to_plot,
        x="abs_thermal_efficiency_change",
        y="combined_name",
        hue="component_name",
        kind="box",
        log_scale=True,
        order=[" ".join(entry[0]) for entry in reversed(sorted_medians)],
        whis=(1, 99),
        flierprops={"marker": "D", "alpha": 0.7},
        hue_order=(alphabetical_component_names:=sorted(set(frame_to_plot["component_name"]))),
        height=32/5,
        aspect=48/32,
        legend_out=True,
    )
    # gng.figure.axes[0].tick_params(axis="x", rotation=90)
    gng.figure.axes[0].set_xlabel("Absolute fractional change in thermal efficiency")
    gng.figure.axes[0].set_ylabel("")
    gng.figure.axes[0].axvline(0.01, linestyle="--", color="grey", label="1% impact")
    gng.figure.axes[0].axvline(0.042, linestyle="-.", color="grey", label="4.2% impact")
    gng.legend.set_title("Component")
    gng.legend.set_alpha(1)
    gng.figure.axes[0].set_xlim((min_ylim:=10**(-15)), 10**3)
    # gng.figure.axes[0].axhspan(min_ylim, 0.042,     color="grey",
    # zorder=0,
    # label="Rejected parameters", alpha=0.1)
    # gng.figure.axes[0].axhspan(0.01, 0.042,     color="C0",
    # zorder=0,
    # label="Limited impact", alpha=0.1)
    # sns.move_legend(gng.figure, "upper left", bbox_to_anchor=(0.1, 0.725))
    # rect = mpatches.Rectangle(
    #     [ax.get_position().x0, ax.get_position().y0],
    #     ax.get_position().x1 - ax.get_position().x0,
    #     0.5,
    #     ec="k",
    #     fc="grey",
    #     alpha=0.1,
    #     clip_on=False,
    #     transform=fig.transFigure,
    #     linewidth=0,
    # )
    # ax.add_patch(rect)
    plt.savefig(
        f"abs_box_sensitivity_autotherm_G_{G}_{INDEX}.png",
        transparent=True,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    gng = sns.catplot(
        frame_to_plot,
        x="abs_thermal_efficiency_change",
        y="combined_name",
        hue="component_name",
        kind="boxen",
        log_scale=True,
        order=[" ".join(entry[0]) for entry in reversed(sorted_medians)],
        outlier_prop=0,
        showfliers=False,
        hue_order=alphabetical_component_names,
        height=32/5,
        aspect=48/32,
        legend_out=True,
    )
    # gng.figure.axes[0].tick_params(axis="x", rotation=90)
    gng.figure.axes[0].set_xlabel("Absolute fractional change in thermal efficiency")
    gng.figure.axes[0].set_ylabel("")
    gng.figure.axes[0].axvline(0.01, linestyle="--", color="grey", label="1% impact")
    gng.figure.axes[0].axvline(0.042, linestyle="-.", color="grey", label="4.2% impact")
    gng.legend.set_title("Component")
    gng.legend.set_alpha(1)
    gng.figure.axes[0].set_xlim((min_ylim:=10**(-15)), 10**3)
    # gng.figure.axes[0].axhspan(min_ylim, 0.042,     color="grey",
    # zorder=0,
    # label="Rejected parameters", alpha=0.1)
    # gng.figure.axes[0].axhspan(0.01, 0.042,     color="C0",
    # zorder=0,
    # label="Limited impact", alpha=0.1)
    # sns.move_legend(gng.figure, "upper left", bbox_to_anchor=(0.1, 0.725))
    plt.savefig(
        f"abs_boxen_sensitivity_autotherm_G_{G}_{INDEX}.png",
        transparent=True,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    # Violin plot
    # gng = sns.catplot(
    #     frame_to_plot,
    #     x="combined_name",
    #     y="abs_thermal_efficiency_change",
    #     hue="component_name",
    #     kind="violin",
    #     log_scale=True,
    #     order=[" ".join(entry[0]) for entry in sorted_medians],
    # )
    # ax = plt.gca()
    # ax.tick_params(axis="x", rotation=90)
    # ax.set_xlabel("")
    # ax.set_ylabel("Absolute fractional change in thermal efficiency")
    # gng.figure.axes[0].axhline(0.01, linestyle="--", color="grey", label="1% impact")
    # gng.figure.axes[0].axhline(0.042, linestyle="-.", color="grey", label="4.2% impact")
    # gng.legend.set_title("Component")
    # plt.savefig(
    #     f"abs_violin_sensitivity_autotherm_G_{G}_{INDEX}.png",
    #     transparent=True,
    #     format="png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # Plot the median values only as a scatter
    median_values_only = {combined_name: sub_frame["abs_thermal_efficiency_change"].median() for combined_name, sub_frame in frame_to_plot.groupby("combined_name")}
    fig = plt.figure(figsize=(32 / 5, 32 / 5))
    ax = plt.gca()
    sns.scatterplot(
        y = reversed(sorted_names:=[" ".join(entry[0]) for entry in sorted_medians]),
        x=[median_values_only[name] for name in reversed(sorted_names)],
        hue=[entry[0][0] for entry in sorted_medians],
        ax=ax,
        marker="D",
        s=100,
        alpha=1,
        hue_order=alphabetical_component_names,
        edgecolor=None,
    )
    # ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("")
    ax.set_xlabel("Median absolute fractional change in thermal efficiency")
    ax.set_xscale("log")
    ax.axvline(0.01, linestyle="--", color="grey", label="1% impact")
    ax.axvline(0.042, linestyle="-.", color="grey", label="4.2% impact")
    ax.set_xlim((min_ylim:=10**(-4)), 1/3)
    ax.axvspan(min_ylim, 0.042,     color="grey",
    zorder=0,
    label="Rejected parameters", alpha=0.1)
    ax.axvspan(0.01, 0.042,     color="C0",
    zorder=0,
    label="Limited impact", alpha=0.1)
    legend = ax.legend(fancybox=True, framealpha=1, facecolor="white", bbox_to_anchor=(1, 1))
    # legend.set_title("Component")
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1)
    plt.savefig(
        f"abs_scatter_sensitivity_autotherm_G_{G}_{INDEX}.png",
        transparent=True,
        format="png",
        dpi=400,
        bbox_inches="tight",
    )

# Output, for each variable, the median impact.
mapping: dict[int, dict[str, float]] = {}
for G in tqdm([200, 400, 600, 800, 1000]):
    parameter_median_squared_fractional_changes: dict[tuple[str, str], float] = {}
    parameter_mean_squared_fractional_changes: dict[tuple[str, str], float] = {}
    frame_to_plot = combined_output_frame[
        combined_output_frame["solar_irradiance"] == G
    ]
    for component_name, sub_frame in frame_to_plot.groupby("component_name"):
        for parameter_name, sub_sub_frame in sub_frame.groupby("parameter_name"):
            parameter_median_squared_fractional_changes[
                component_name, parameter_name
            ] = float(sub_sub_frame["abs_thermal_efficiency_change"].median())
            parameter_mean_squared_fractional_changes[
                component_name, parameter_name
            ] = float(sub_sub_frame["abs_thermal_efficiency_change"].median())
    sorted_medians = sorted(
        parameter_median_squared_fractional_changes.items(), key=lambda item: item[1]
    )
    median_values_only = {combined_name: sub_frame["abs_thermal_efficiency_change"].median() for combined_name, sub_frame in frame_to_plot.groupby("combined_name")}
    print("\n".join([f"{'**' if median_values_only[name] >= 0.042 else '*' if median_values_only[name] >= 0.01 else ''}{name}: {median_values_only[name]:.3g}" for name in [f"{entry[0][0]} {entry[0][1]}" for entry in sorted_medians]]))
    mapping[G] = {name: 100 * float(f"{median_values_only[name]:.3g}") for name in  [f"{entry[0][0]} {entry[0][1]}" for entry in sorted_medians]}

frame = pd.DataFrame.from_dict(mapping)
frame = frame.sort_values(1000, ascending=False)

#####################################################################
# Plot a single variable with the parameter value dictating the hue #
#####################################################################

sns.set_context("notebook")

parameter_name_to_unit_map: dict[str, str] = {
    "Absorptivity": None,
    "Collector width": "pipes",
    "Density": "kg/m$^3$",
    "Diffuse reflection coefficient": None,
    "Emissivity": None,
    "Heat capacity": "J/kgK",
    "Reference efficiency": None,
    "Thermal coefficient": "K$^-1$",
    "Thermal conductivity": "W/m K",
    "Thickness": "m",
    "Transmissivity": None,
    "Width": "m",
}

start = 0
end = 8


def _size_from_component_and_parameter(component: str, parameter: str) -> int:
    """Determine the size of the points based on the component and parameter."""
    # if component == "Absorber":
    #     if parameter in ("Thickness"):
    #         return 0.5
    # if component in ("Adhesive", "Eva", "Tedlar", "Air gap",):
    #     if parameter in ("Thermal conductivity"):
    #         return 0.5
    # if component == "Bond":
    #     if parameter in ("Thermal conductivity", "Width"):
    #         return 0.5
    # if component == "Glass":
    #     if parameter in ("Diffuse reflection coefficient", "Emissivity", "Thermal conductivity", "Thickenss"):
    #         return 0.5
    # if component == "Insulation":
    #     if parameter in ("Thermal conductivity", "Thickness"):
    #         return 0.5
    # if component == "PV":
    #     if parameter in ("Emissivity", "Reference efficiency", "Thermal coefficienct", "Transmissivity"):
    #         return 0.5
    # if component == "Pvt":
    #     return 0.5
    return 0.25


for _component_name, start_hue in zip(
    [
        # "Absorber",
        "Adhesive",
        "Air gap",
        # "Bond",
        "Eva",
        "Glass",
        "Insulation",
        "Pv",
        "Pvt",
        "Tedlar",
    ][start:end],
    [0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4][start:end],
):
    for _parameter_name in set(
        combined_output_frame[(combined_output_frame["component_name"] == _component_name)][
            "parameter_name"
        ]
    ):
        frame_to_plot = combined_output_frame[
            (combined_output_frame["component_name"] == _component_name)
            & (combined_output_frame["parameter_name"] == _parameter_name)
            & (
                ~combined_output_frame["parameter_value"]
                .astype(str)
                .str.contains("checkpoint")
            )
            & (combined_output_frame["solar_irradiance"] > 0)
        ]
        # Drop nan entries
        frame_to_plot = frame_to_plot.dropna(subset="percentage_thermal_efficiency_change")
        frame_to_plot.loc[:, "parameter_value"] = frame_to_plot[
            "parameter_value"
        ].astype(float)
        if _component_name == "PV-T":
            frame_to_plot.loc[:, "parameter_value"] = [
                int(0.86 / entry)
                for entry in frame_to_plot["parameter_value"].astype(float)
            ]
        _unit: str | None = parameter_name_to_unit_map[_parameter_name]
        # sns.swarmplot(
        #     frame_to_plot,
        #     x="percentage_thermal_efficiency_change",
        #     y="parameter_name",
        #     hue="parameter_value",
        #     palette=(
        #         this_palette := sns.cubehelix_palette(
        #             start=start_hue,
        #             rot=-0.2,
        #             dark=0.1,
        #             n_colors=len(set(frame_to_plot["parameter_value"])),
        #         )
        #     ),
        #     marker="D",
        #     size=_size_from_component_and_parameter(_component_name, _parameter_name),
        # )
        # Create the data
        # rs = np.random.RandomState(1979)
        # x = rs.randn(500)
        # g = np.tile(list("ABCDEFGHIJ"), 50)
        # df = pd.DataFrame(dict(x=x, g=g))
        # m = df.g.map(ord)
        # df["x"] += m
        # Initialize the FacetGrid object
        # pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(frame_to_plot, row="parameter_value", hue="parameter_value", aspect=15, height=0.5, palette=(
                this_palette := sns.cubehelix_palette(
                    start=start_hue,
                    rot=-0.6,
                    dark=0.15,
                    light=0.9,
                    n_colors=len(set(frame_to_plot["parameter_value"])),
                )
            ),)
        # Draw the densities in a few steps
        g.map(sns.kdeplot, "percentage_thermal_efficiency_change",
            bw_adjust=.5, clip_on=False,
            fill=True, alpha=1, linewidth=1.5)
        # g.map(sns.kdeplot, "percentage_thermal_efficiency_change",
        #     bw_adjust=.5, clip_on=False, fill=True,
        #     linewidth=1.5,
        #     marker="D",
        #     alpha=0.1,
        #     size=10
        # )
        g.map(sns.kdeplot, "percentage_thermal_efficiency_change", clip_on=False, color="w", lw=2, bw_adjust=.5)
        # g.map(sns.stripplot, "percentage_thermal_efficiency_change",
        #     #   clip_on=False,
        #         color="w", linewidth=2,
        #         # bw_adjust=.5
        #     )
        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, 0.8, f"{label} W/m$^2$", fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes) \

        g.map(label, "parameter_value")
        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-0.1)
        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        # g.set_xlabels("Percentage thermal efficiency change")
        # axis = plt.gca()
        # levels = sorted(set(frame_to_plot["parameter_value"]))
        # cmap, norm = mcolors.from_levels_and_colors(levels, this_palette[:-1])
        # # norm = plt.Normalize(
        # #     frame_to_plot["parameter_value"].min(),
        # #     frame_to_plot["parameter_value"].max(),
        # # )
        # scalar_mappable = plt.cm.ScalarMappable(
        #     cmap=cmap,
        #     norm=norm,
        # )
        # colorbar = axis.figure.colorbar(
        #     scalar_mappable,
        #     ax=axis,
        #     label=(
        #         (
        #             f"{_component_name} {_parameter_name}"
        #             if _component_name != "PV-T"
        #             else "Number of pipes"
        #         )
        #         + (f" / {_unit}" if _unit is not None else "")
        #     ),
        # )
        axis = plt.gca()
        # Stripplot
        # sns.stripplot(
        #     frame_to_plot,
        #     x="solar_irradiance",
        #     y="percentage_thermal_efficiency_change",
        #     hue="parameter_value",
        #     palette=(
        #         this_palette := sns.cubehelix_palette(
        #             start=start_hue,
        #             rot=-0.6,
        #             dark=0.15,
        #             light=0.9,
        #             n_colors=len(set(frame_to_plot["parameter_value"])),
        #         )
        #     ),
        #     marker="D",
        #     jitter=1,
        #     dodge=True,
        #     orient="h",
        #     alpha=0.3,
        #     size=10,
        # )
        # axis = plt.gca()
        # norm = plt.Normalize(
        #     frame_to_plot["parameter_value"].min(),
        #     frame_to_plot["parameter_value"].max(),
        # )
        # scalar_mappable = plt.cm.ScalarMappable(
        #     cmap=mcolors.LinearSegmentedColormap.from_list(
        #         "Custom",
        #         [(value, colour) for value, colour in zip(this_palette.as_hex(), sorted(frame_to_plot["parameter_value"]))],
        #     ),
        #     norm=norm,
        # )
        # colorbar = axis.figure.colorbar(
        #     scalar_mappable,
        #     ax=axis,
        #     label=(
        #         (
        #             f"{_component_name} {_parameter_name}"
        #             if _component_name != "PV-T"
        #             else "Number of pipes"
        #         )
        #         + (f" / {_unit}" if _unit is not None else "")
        #     ),
        # )
        # axis = plt.gca()
        # axis.get_legend().remove()
        # axis.set_xlabel("Percentage change in thermal efficiency /  %")
        # axis.set(ylabel=None, yticklabels=[])
        # axis.axvline(x=0, linestyle="--", color="grey", linewidth=1)
        plt.savefig(
            f"(KDE) Sensitivity of {_component_name} {_parameter_name}_{INDEX}.png",
            transparent=True,
            dpi=400,
            bbox_inches="tight",
            pad_inches=0,
        )
        # plt.show()

for _component_name, start_hue in zip(
    [
        # "Absorber",
        "Adhesive",
        "Air gap",
        # "Bond",
        "Eva",
        "Glass",
        "Insulation",
        "Pv",
        "Pvt",
        "Tedlar",
    ][start:end],
    [0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4][start:end],
):
    for _parameter_name in set(
        combined_output_frame[
            (combined_output_frame["component_name"] == _component_name)
        ]["parameter_name"]
    ):
        frame_to_plot = combined_output_frame[
            (combined_output_frame["component_name"] == _component_name)
            & (combined_output_frame["parameter_name"] == _parameter_name)
            & (
                ~combined_output_frame["parameter_value"]
                .astype(str)
                .str.contains("checkpoint")
            )
        ]
        frame_to_plot.loc[:, "parameter_value"] = frame_to_plot[
            "parameter_value"
        ].astype(float)
        if _component_name == "Pvt":
            frame_to_plot.loc[:, "parameter_value"] = [
                int(0.86 / entry)
                for entry in frame_to_plot["parameter_value"].astype(float)
            ]
        _unit: str | None = parameter_name_to_unit_map[_parameter_name]
        fig = plt.figure(figsize=(48 / 5, 32 / 5))
        sns.swarmplot(
            frame_to_plot,
            x="percentage_thermal_efficiency_change",
            y="parameter_name",
            hue="parameter_value",
            palette=(
                this_palette := sns.cubehelix_palette(
                    start=start_hue,
                    rot=-0.2,
                    dark=0.1,
                    n_colors=len(set(frame_to_plot["parameter_value"])),
                )
            ),
            marker="D",
            size=_size_from_component_and_parameter(_component_name, _parameter_name),
        )
        # sns.stripplot(
        #     frame_to_plot,
        #     x="percentage_thermal_efficiency_change",
        #     y="parameter_name",
        #     hue="parameter_value",
        #     palette=(
        #         this_palette := sns.cubehelix_palette(
        #             start=start_hue,
        #             rot=-0.2,
        #             dark=0.1,
        #             n_colors=len(set(frame_to_plot["parameter_value"])),
        #         )
        #     ),
        #     marker="D",
        #     alpha=0.1,
        #     size=5,
        # )
        axis = plt.gca()
        norm = plt.Normalize(
            frame_to_plot["parameter_value"].min(),
            frame_to_plot["parameter_value"].max(),
        )
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "Custom",
                this_palette.as_hex(),
                len(set(frame_to_plot["parameter_value"])),
            ),
            norm=norm,
        )
        colorbar = axis.figure.colorbar(
            scalar_mappable,
            ax=axis,
            label=(
                (
                    f"{_component_name} {_parameter_name}"
                    if _component_name != "Pvt"
                    else "Number of pipes"
                )
                + (f" / {_unit}" if _unit is not None else "")
            ),
        )
        axis = plt.gca()
        axis.get_legend().remove()
        axis.set_xlabel("Percentage change in thermal efficiency /  %")
        axis.set(ylabel=None, yticklabels=[])
        axis.axvline(x=0, linestyle="--", color="grey", linewidth=1)
        plt.savefig(
            f"(Swarm) Sensitivity of {_component_name} {_parameter_name}_{INDEX}.png",
            transparent=True,
            dpi=400,
            bbox_inches="tight",
            pad_inches=0,
        )

################################
# Circular plot of sensitivity #
################################

# Expand the thesis blue-first palette to have double the number of colours
blue_first_thesis_colours = blue_first_thesis_palette.as_hex()

colourmap = mcolors.LinearSegmentedColormap.from_list(
    "", np.array(blue_first_thesis_colours), 20
)

blue_first_thesis_palette_18 = sns.color_palette(
    colourmap(np.linspace(0, 1, 18)).tolist()
)
sns.set_palette(blue_first_thesis_palette_18)

blue_first_thesis_palette_20 = sns.color_palette(
    colourmap(np.linspace(0, 1, 20)).tolist()
)
sns.set_palette(blue_first_thesis_palette_20)

sectors = {
    component_name: len(
        set(
            combined_output_frame[
                combined_output_frame["component_name"] == component_name
            ]["parameter_name"]
        )
    )
    for component_name in {
        entry
        for entry in set(combined_output_frame["component_name"])
        if entry != "Pipe"
    }
}
circos = Circos(sectors, space=5)

for sector in circos.sectors:
    # Plot sector name
    sector.text(f"{sector.name}", r=110, size=15)
    # # Create x positions & randomized y values for data plotting
    # x = np.arange(sector.start, sector.end) + 0.5
    # y = np.random.randint(0, 100, len(x))
    # # Plot line
    # line_track = sector.add_track((75, 100), r_pad_ratio=0.1)
    # line_track.axis()
    # line_track.xticks_by_interval(1)
    # line_track.line(x, y)
    # # Plot points
    # points_track = sector.add_track((45, 70), r_pad_ratio=0.1)
    # points_track.axis()
    # points_track.scatter(x, y)
    # Plot bar
    bar_track = sector.add_track((15, 100), r_pad_ratio=0.1)
    bar_track.axis()
    category_names = set(
        combined_output_frame[combined_output_frame["component_name"] == sector.name][
            "parameter_name"
        ]
    )
    name_to_position_map = {
        name: index / len(category_names) * (sector.end - sector.start)
        + sector.start
        + 0.5
        for index, name in enumerate(category_names)
    }
    # bar_track.xticks(sorted(name_to_position_map.values()), label_size=8, label_orientation="vertical", label_formatter=lambda value:{v: k for k, v in name_to_position_map.items()}[value + .5])
    bar_track.scatter(
        x=[
            name_to_position_map[entry]
            for entry in combined_output_frame[
                (combined_output_frame["component_name"] == sector.name)
                & (combined_output_frame["squared_thermal_efficiency_change"].notna())
            ]["parameter_name"]
        ],
        y=list(
            combined_output_frame[
                combined_output_frame["component_name"] == sector.name
            ]["squared_thermal_efficiency_change"].dropna()
        ),
        marker="D",
        s=20,
        alpha=0.3,
    )

circos.savefig(f"circular_sensitivity_{INDEX}.png", dpi=400)

circos = Circos(sectors, space=5)
max_mean: float = 0
# Create a bar with the mean value
for sector in circos.sectors:
    # Plot sector name
    sector.text(f"{sector.name}", r=110, size=15)
    # # Create x positions & randomized y values for data plotting
    # x = np.arange(sector.start, sector.end) + 0.5
    # y = np.random.randint(0, 100, len(x))
    # # Plot line
    # line_track = sector.add_track((75, 100), r_pad_ratio=0.1)
    # line_track.axis()
    # line_track.xticks_by_interval(1)
    # line_track.line(x, y)
    # # Plot points
    # points_track = sector.add_track((45, 70), r_pad_ratio=0.1)
    # points_track.axis()
    # points_track.scatter(x, y)
    # Plot bar
    bar_track = sector.add_track((15, 100), r_pad_ratio=0.1)
    bar_track.axis()
    # bar_track.xticks(sorted(name_to_position_map.values()), label_size=8, label_orientation="vertical", label_formatter=lambda value:{v: k for k, v in name_to_position_map.items()}[value + .5])
    bar_data = {
        parameter_name: sub_frame["squared_thermal_efficiency_change"].mean()
        for parameter_name, sub_frame in combined_output_frame[
            combined_output_frame["component_name"] == sector.name
        ].groupby("parameter_name")
    }
    name_to_position_map = {
        name: index / len(bar_data.keys()) * (sector.end - sector.start)
        + sector.start
        + 0.5
        for index, name in enumerate(bar_data.keys())
    }
    x_data = [name_to_position_map[key] for key in sorted(bar_data)]
    y_data = [bar_data[key] for key in sorted(bar_data)]
    bar_track.bar(
        x=x_data,
        height=y_data,
    )
    print(f"Component {sector.name}")
    print("\n".join(sorted(bar_data)))
    max_mean = max(max_mean, max(y_data))

circos.savefig(f"circular_sensitivity_bar_{INDEX}.png", dpi=400)

vmax = combined_output_frame["squared_thermal_efficiency_change"].dropna().max()

sectors = {
    component_name: len(
        set(
            combined_output_frame[
                combined_output_frame["component_name"] == component_name
            ]["parameter_name"]
        )
    )
    for component_name in {
        entry
        for entry in set(combined_output_frame["component_name"])
        if entry != "Pipe"
    }
}
circos = Circos(sectors, space=5)

for sector in circos.sectors:
    # Plot sector name
    sector.text(f"{sector.name}", r=110, size=15)
    # # Create x positions & randomized y values for data plotting
    # x = np.arange(sector.start, sector.end) + 0.5
    # y = np.random.randint(0, 100, len(x))
    # # Plot line
    # line_track = sector.add_track((75, 100), r_pad_ratio=0.1)
    # line_track.axis()
    # line_track.xticks_by_interval(1)
    # line_track.line(x, y)
    # # Plot points
    # points_track = sector.add_track((45, 70), r_pad_ratio=0.1)
    # points_track.axis()
    # points_track.scatter(x, y)
    # Plot bar
    bar_track = sector.add_track((15, 100), r_pad_ratio=0.1)
    bar_track.axis()
    category_names = sorted(
        set(
            combined_output_frame[
                combined_output_frame["component_name"] == sector.name
            ]["parameter_name"]
        )
    )
    name_to_position_map = {
        name: index / len(category_names) * (sector.end - sector.start)
        + sector.start
        + 0.5
        for index, name in enumerate(category_names)
    }
    # bar_track.xticks(sorted(name_to_position_map.values()), label_size=8, label_orientation="vertical", label_formatter=lambda value:{v: k for k, v in name_to_position_map.items()}[value + .5])
    bar_track.scatter(
        x=[
            name_to_position_map[entry]
            for entry in combined_output_frame[
                (combined_output_frame["component_name"] == sector.name)
                & (combined_output_frame["squared_thermal_efficiency_change"].notna())
            ]["parameter_name"]
        ],
        y=list(
            combined_output_frame[
                combined_output_frame["component_name"] == sector.name
            ]["squared_thermal_efficiency_change"].dropna()
        ),
        marker="D",
        s=20,
        alpha=0.3,
        vmax=vmax,
    )

circos.savefig(f"circular_sensitivity_with_max_{INDEX}.png", dpi=400)

circos = Circos(sectors, space=5)
# Create a bar with the mean value
for sector in circos.sectors:
    # Plot sector name
    sector.text(f"{sector.name}", r=110, size=15)
    # # Create x positions & randomized y values for data plotting
    # x = np.arange(sector.start, sector.end) + 0.5
    # y = np.random.randint(0, 100, len(x))
    # # Plot line
    # line_track = sector.add_track((75, 100), r_pad_ratio=0.1)
    # line_track.axis()
    # line_track.xticks_by_interval(1)
    # line_track.line(x, y)
    # # Plot points
    # points_track = sector.add_track((45, 70), r_pad_ratio=0.1)
    # points_track.axis()
    # points_track.scatter(x, y)
    # Plot bar
    bar_track = sector.add_track((15, 100), r_pad_ratio=0.1)
    bar_track.axis()
    # bar_track.xticks(sorted(name_to_position_map.values()), label_size=8, label_orientation="vertical", label_formatter=lambda value:{v: k for k, v in name_to_position_map.items()}[value + .5])
    bar_data = {
        parameter_name: sub_frame["squared_thermal_efficiency_change"].mean()
        for parameter_name, sub_frame in combined_output_frame[
            combined_output_frame["component_name"] == sector.name
        ].groupby("parameter_name")
    }
    name_to_position_map = {
        name: index / len(bar_data.keys()) * (sector.end - sector.start)
        + sector.start
        + 0.5
        for index, name in enumerate(bar_data.keys())
    }
    x_data = [name_to_position_map[key] for key in sorted(bar_data)]
    y_data = [bar_data[key] for key in sorted(bar_data)]
    bar_track.bar(
        x=x_data,
        height=y_data,
        vmax=max_mean,
    )

circos.savefig(f"circular_sensitivity_bar_with_max_{INDEX}.png", dpi=400)

########################################
# Plot the impact of the PV efficiency #
########################################

frame_to_plot = combined_output_frame[
    combined_output_frame["parameter_name"] == "Reference efficiency"
]

frame_to_plot["marker"] = [
    "h" if row[1]["wind_speed"] == 1 else "H" for row in frame_to_plot.iterrows()
]

fig = plt.figure(figsize=(48 / 5, 32 / 5))
for marker, sub_frame in frame_to_plot.groupby("marker"):
    sns.set_palette(
        this_palette := (
            sns.cubehelix_palette(
                start=0, rot=-0.2, n_colors=len(set(sub_frame["parameter_value"]))
            )
            if marker == "h"
            else sns.cubehelix_palette(
                start=-0.4, rot=-0.2, n_colors=len(set(sub_frame["parameter_value"]))
            )
        )
    )
    plt.scatter(
        [],
        [],
        alpha=0,
        label=f"Wind speed = {'1' if marker == 'h' else '10'} m/s",
        marker=marker,
    )
    sns.scatterplot(
        sub_frame,
        x="reduced_collector_temperature",
        y="thermal_efficiency",
        hue="parameter_value",
        marker=marker,
        alpha=0.4,
        s=85,
        palette=this_palette,
        linewidth=0,
    )


plt.xlabel("Reduced collector temperature / Km$^2$/W")
plt.ylabel("Thermal efficiency")
axis = plt.gca()

for index, _legend_handler in enumerate(axis.get_legend().legendHandles):
    _legend_handler.set_alpha(1)
    _legend_handler._sizes = [150, 150]

axis.legend(
    [axis.get_legend().legendHandles[1], axis.get_legend().legendHandles[13]],
    ["1 m/s", "10 m/s"],
    title="Wind speed",
    fontsize="medium",
    title_fontsize="medium",
)

for index, _legend_handler in enumerate(axis.get_legend().legendHandles):
    _legend_handler.set_alpha(1)
    _legend_handler._sizes = [150, 150]

norm = plt.Normalize(
    frame_to_plot["parameter_value"].min(),
    frame_to_plot["parameter_value"].max(),
)
scalar_mappable = plt.cm.ScalarMappable(
    cmap=mcolors.LinearSegmentedColormap.from_list(
        "Custom",
        sns.cubehelix_palette(
            start=-0.2, rot=-0.2, n_colors=len(set(frame_to_plot["parameter_value"]))
        ).as_hex(),
        len(set(frame_to_plot["parameter_value"])),
    ),
    norm=norm,
)
colorbar = axis.figure.colorbar(scalar_mappable, ax=axis, label="Electrical efficiency")

scalar_mappable = plt.cm.ScalarMappable(
    cmap=mcolors.LinearSegmentedColormap.from_list(
        "Custom",
        sns.cubehelix_palette(
            start=0, rot=-0.2, n_colors=len(set(frame_to_plot["parameter_value"]))
        ).as_hex(),
        len(set(frame_to_plot["parameter_value"])),
    ),
    norm=norm,
)
colorbar = axis.figure.colorbar(scalar_mappable, ax=axis, label="Electrical efficiency")

# axis.legend(
#     [axis.get_legend().legendHandles[0], axis.get_legend().legendHandles[12]],
#     ["1 m/s", "10 m/s"],
#     title="Wind speed",
#     fontsize="medium",
#     title_fontsize="medium",
# )

# for index, _legend_handler in enumerate(axis.get_legend().legendHandles):
#     _legend_handler.set_alpha(1)
#     _legend_handler._sizes = [150]

plt.savefig(
    f"electrical_vs_thermal_efficiency_scatter_{INDEX}.png",
    transparent=True,
    bbox_inches="tight",
    dpi=400,
)

# plt.show()

# Plot a scatter with lines

fig = plt.figure(figsize=(48 / 5, 32 / 5))
for marker, sub_frame in frame_to_plot.groupby("marker"):
    sns.set_palette(
        this_palette := sns.cubehelix_palette(
            start=0, rot=-0.2, n_colors=len(set(sub_frame["parameter_value"]))
        )
    )
    plt.scatter(
        [], [], alpha=0, label=f"{'1' if marker == 'h' else '10'} m/s", marker=marker
    )
    sns.scatterplot(
        sub_frame,
        x="reduced_collector_temperature",
        y="thermal_efficiency",
        hue="parameter_value",
        marker=marker,
        alpha=0.2,
        s=85,
        palette=this_palette,
        linewidth=0,
    )
    for index, electrical_efficiency in enumerate(
        sorted(set(sub_frame["parameter_value"]))
    ):
        sub_reduced_temperature_fit = np.poly1d(
            np.polyfit(
                (
                    this_frame := sub_frame[
                        sub_frame["parameter_value"] == electrical_efficiency
                    ]
                )["reduced_collector_temperature"],
                this_frame["thermal_efficiency"],
                2,
            )
        )
        sns.lineplot(
            x=(
                x_linspace := np.linspace(
                    min(this_frame["reduced_collector_temperature"]),
                    max(this_frame["reduced_collector_temperature"]),
                )
            ),
            y=sub_reduced_temperature_fit(x_linspace),
            # color=f"C{index}"
            # dashes=(2, 2),
            # label="Quadratic fit",
            palette=this_palette,
            alpha=0.8,
            linewidth=1,
        )

plt.xlabel("Reduced collector temperature / Km$^2$/W")
plt.ylabel("Thermal efficiency")
axis = plt.gca()

axis.legend(
    [axis.get_legend().legendHandles[0], axis.get_legend().legendHandles[12]],
    ["1 m/s", "10 m/s"],
    title="Wind speed",
    fontsize="medium",
    title_fontsize="medium",
)

for index, _legend_handler in enumerate(axis.get_legend().legendHandles):
    _legend_handler.set_alpha(1)
    _legend_handler._sizes = [150]

norm = plt.Normalize(
    frame_to_plot["parameter_value"].min(),
    frame_to_plot["parameter_value"].max(),
)
scalar_mappable = plt.cm.ScalarMappable(
    cmap=mcolors.LinearSegmentedColormap.from_list(
        "Custom", this_palette.as_hex(), len(set(frame_to_plot["parameter_value"]))
    ),
    norm=norm,
)
colorbar = axis.figure.colorbar(scalar_mappable, ax=axis, label="Electrical efficiency")

plt.savefig(
    f"electrical_vs_thermal_efficiency_scatter_with_lines_{INDEX}.png",
    transparent=True,
    bbox_inches="tight",
    dpi=400,
)

# plt.show()

# FIXME: Try a category plot
# rot: float = -0.2
# start: float = 0
# start_step: float = 0.2
# for component_name, sub_frame in combined_output_frame.groupby("component_name"):
#     fig = plt.figure(figsize=(48 / 5, 32 / 5))
#     this_palette = sns.cubehelix_palette(start=start, rot=rot)
#     start -= start_step
#     sns.stripplot(
#         sub_frame,
#         x="parameter_name",
#         y="percentage_thermal_efficiency_change",
#         hue="parameter_value",
#         palette=this_palette,
#         dodge=True,
#         marker="D",
#         s=3,
#         alpha=0.5,
#     )
#     # plt.show()

###############################
# Overall plot of sensitivity #
###############################

# Expand the thesis blue-first palette to have double the number of colours
blue_first_thesis_colours = blue_first_thesis_palette.as_hex()

colourmap = mcolors.LinearSegmentedColormap.from_list(
    "", np.array(blue_first_thesis_colours), 20
)

blue_first_thesis_palette_18 = sns.color_palette(
    colourmap(np.linspace(0, 1, 18)).tolist()
)
sns.set_palette(blue_first_thesis_palette_18)

blue_first_thesis_palette_20 = sns.color_palette(
    colourmap(np.linspace(0, 1, 20)).tolist()
)
sns.set_palette(blue_first_thesis_palette_20)

combined_output_frame.loc[
    combined_output_frame["component_name"] == "Pv", "component_name"
] = "PV"
combined_output_frame.loc[
    combined_output_frame["component_name"] == "Pvt", "component_name"
] = "PV-T"

sns.set_context("talk")

# fig, axes = plt.subplots(1, 1, figsize=(64 / 5, 32 / 5))
# A separate case is needed for the pipe diameters
order_by_square = (
    (
        frame_to_plot := combined_output_frame[
            combined_output_frame["component_name"] != "Pipe"
        ]
    )
    .groupby(by=["parameter_name"])["squared_thermal_efficiency_change"]
    .mean()
    .sort_values()
    .iloc[::-1]
    .index
)

# Bar facet
grid = sns.catplot(
    data=frame_to_plot,
    x="percentage_thermal_efficiency_change",
    y="parameter_name",
    col="component_name",
    hue="component_name",
    height=10,
    aspect=0.5,
    kind="bar",
    errorbar=None,
    order=order_by_square,
)
grid.set_axis_labels("Percentage change", "Parameter")
grid.set_titles("{col_name}".capitalize(), weight="bold")
grid.despine(left=True)
for ax in grid.axes.flat:
    ax.axvline(x=0, linestyle="--", color="grey", linewidth=1)

grid.legend.remove()
# grid.add_legend(title="Component")

plt.savefig(
    f"overall_sensitivity_bar_{INDEX}.png", transparent=True, bbox_inches="tight", dpi=600
)

# plt.show()

# Distribution facet
grid = sns.catplot(
    data=frame_to_plot,
    x="percentage_thermal_efficiency_change",
    y="parameter_name",
    col="component_name",
    hue="component_name",
    height=10,
    aspect=0.5,
    kind="boxen",
    errorbar=None,
    order=order_by_square,
    k_depth="full",
)
grid.set_axis_labels("Percentage change", "Parameter")
grid.set_titles("{col_name}".capitalize(), weight="bold")
grid.despine(left=True)
for ax in grid.axes.flat:
    ax.axvline(x=0, linestyle="--", color="grey", linewidth=1)

grid.legend.remove()

plt.savefig(
    f"overall_sensitivity_boxen_{INDEX}.png", transparent=True, bbox_inches="tight", dpi=600
)

# plt.show()

# Distribution facet
grid = sns.catplot(
    data=frame_to_plot,
    x="percentage_thermal_efficiency_change",
    y="parameter_name",
    col="component_name",
    hue="component_name",
    height=10,
    aspect=0.5,
    kind="box",
    errorbar=None,
    order=order_by_square,
    whis=(0, 100),
    gap=0,
    dodge=False,
)
grid.set_axis_labels("Percentage change", "Parameter")
grid.set_titles("{col_name}".capitalize(), weight="bold")
grid.despine(left=True)
for ax in grid.axes.flat:
    ax.axvline(x=0, linestyle="--", color="grey", linewidth=1)

grid.legend.remove()

plt.savefig(
    f"overall_sensitivity_box_and_whisker_{INDEX}.png",
    transparent=True,
    bbox_inches="tight",
    dpi=600,
)

# plt.show()

# Scatter facet
grid = sns.catplot(
    data=frame_to_plot,
    x="percentage_thermal_efficiency_change",
    y="parameter_name",
    col="component_name",
    hue="component_name",
    height=10,
    aspect=0.5,
    kind="strip",
    errorbar=None,
    order=order_by_square,
    marker="D",
    alpha=0.3,
    s=65,
    dodge=False,
    jitter=False,
)
grid.set_axis_labels("Percentage change", "Parameter")
grid.set_titles("{col_name}".capitalize(), weight="bold")
grid.despine(left=True)
for ax in grid.axes.flat:
    ax.axvline(x=0, linestyle="--", color="grey", linewidth=1)

grid.legend.remove()

plt.savefig(
    f"overall_sensitivity_strip_{INDEX}.png", transparent=True, bbox_inches="tight", dpi=600
)

# plt.show()

sns.set_context("notebook")


# sns.barplot(
#     frame_to_plot,
#     x="fractional_thermal_efficiency_change",
#     y="parameter_name",
#     ax=(axis := axes),
#     linewidth=1,
#     order=order_by_square,
#     palette=blue_first_thesis_palette_18,
#     hue="component_name",
#     errorbar=None,
# )
# sns.boxplot(
#     frame_to_plot,
#     x="fractional_thermal_efficiency_change",
#     y="parameter_name",
#     ax=(axis := axes),
#     linewidth=1,
#     order=order_by_square,
#     palette=blue_first_thesis_palette_18,
#     hue="component_name",
#     whis=(0, 100),
#     gap=0,
#     dodge=False,
# )
# sns.stripplot(
#     frame_to_plot,
#     x="fractional_thermal_efficiency_change",
#     y="parameter_name",
#    palette=blue_first_thesis_palette_18,
#     ax=axis,
#     alpha=0.8,
#     linewidth=0,
#     marker="D",
#     size=1,
#     dodge=False,
#     # jitter=True,
#     order=order_by_square,
#     hue="component_name",
#     label=None,
# )
# plt.xlabel("Fractional change in thermal efficiency")
# plt.ylabel("Parameter name")
# Remove top and right axes, keep bottom and left with ticks
# axis.spines["top"].set_visible(False)
# axis.spines["right"].set_visible(False)
# axis.tick_params(bottom=True, left=True)  # Enable ticks on bottom and left axes
# Adjust margins (optional, adjust values for desired spacing)
# fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
# plt.show()

#########################################
# Individual plots for component layers #
#########################################

pipe_diameter_regex = re.compile(r"Diameter inner (?P<inner_diameter>.*) outer")

# Plot the fractional, squared, and percentage changes
# Determine the range to used based on the extreme range
vmax = combined_output_frame["fractional_thermal_efficiency_change"].dropna().max()
vmin = combined_output_frame["fractional_thermal_efficiency_change"].dropna().min()

sns.set_palette(blue_first_thesis_palette)
sns.set_style("ticks")

for component_name, sub_frame in combined_output_frame.groupby("component_name"):
    fig, axes = plt.subplots(1, 1, figsize=(48 / 5, 32 / 5))
    # A separate case is needed for the pipe diameters
    plt.axvline(x=0, linestyle="--", color="grey", linewidth=1)
    if component_name == "Pipe":
        # Create a new frame containing the inner diameter and outer diameter values
        pipe_frame = sub_frame.copy()
        pipe_frame["outer_diameter"] = pipe_frame["parameter_value"].astype(float)
        pipe_frame["inner_diameter"] = None
        for index, row in pipe_frame.iterrows():
            pipe_frame.at[index, "inner_diameter"] = float(
                pipe_diameter_regex.match(row["parameter_name"]).group("inner_diameter")
            )
        pipe_frame["thickness"] = (
            pipe_frame["outer_diameter"] - pipe_frame["inner_diameter"]
        ).round(2)
        axis = axes
        sns.scatterplot(
            pipe_frame,
            x="fractional_thermal_efficiency_change",
            y="outer_diameter",
            hue="thickness",
            marker="D",
            s=50,
            linewidth=0,
            alpha=0.2,
            palette=(
                this_palette := sns.cubehelix_palette(start=0, rot=-0.2, n_colors=14)
            ),
        )
        norm = plt.Normalize(
            pipe_frame["fractional_thermal_efficiency_change"].min(),
            pipe_frame["fractional_thermal_efficiency_change"].max(),
        )
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "Custom", this_palette.as_hex(), len(set(pipe_frame["inner_diameter"]))
            ),
            norm=norm,
        )
        colorbar = axis.figure.colorbar(
            scalar_mappable, ax=axis, label="Inner pipe diameter / mm"
        )
        axis.get_legend().remove()
        plt.xlabel("Fractional change in thermal efficiency")
        plt.ylabel("Outer pipe diameter / mm")
    else:
        order_by_square = (
            sub_frame.groupby(by=["parameter_name"])[
                "squared_thermal_efficiency_change"
            ]
            .mean()
            .sort_values()
            .iloc[::-1]
            .index
        )
        if (num_parameter_names := len(set(sub_frame["parameter_name"]))) > len(
            sns.color_palette()
        ):
            norm = plt.Normalize(0, num_parameter_names)
            scalar_mappable = plt.cm.ScalarMappable(
                cmap=mcolors.LinearSegmentedColormap.from_list(
                    "Custom", sns.color_palette().as_hex(), num_parameter_names
                ),
                norm=norm,
            )
        sns.violinplot(
            sub_frame,
            x="fractional_thermal_efficiency_change",
            y="parameter_name",
            ax=(axis := axes),
            cut=0,
            inner=None,
            linewidth=1,
            order=order_by_square,
            hue_order=order_by_square,
            palette=blue_first_thesis_palette,
            hue="parameter_name",
            alpha=0.8,
        )
        sns.stripplot(
            sub_frame,
            x="fractional_thermal_efficiency_change",
            y="parameter_name",
            palette=blue_first_thesis_palette,
            hue_order=order_by_square,
            ax=axis,
            alpha=0.2,
            linewidth=0,
            marker="D",
            size=3,
            dodge=False,
            jitter=False,
            order=order_by_square,
        )
        plt.xlabel("Fractional change in thermal efficiency")
        plt.ylabel("Parameter name")
    # Remove top and right axes, keep bottom and left with ticks
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(bottom=True, left=True)  # Enable ticks on bottom and left axes
    # Adjust margins (optional, adjust values for desired spacing)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.xlim(round(10 * vmin, -1) / 10, round(10 * vmax, 1) / 10)
    plt.title(
        (
            "PV-T"
            if component_name == "Pvt"
            else "PV" if component_name == "Pv" else component_name.capitalize()
        ),
        weight="bold",
    )
    plt.savefig(
        f"fractions_violins_for_{component_name}_with_border_{INDEX}.png",
        transparent=True,
        dpi=400,
        bbox_inches="tight",
    )

plt.close()

# Determine the range to used based on the extreme range
vmax = combined_output_frame["squared_thermal_efficiency_change"].dropna().max()
vmin = combined_output_frame["squared_thermal_efficiency_change"].dropna().min()

sns.set_palette(blue_first_thesis_palette)
sns.set_style("ticks")

for component_name, sub_frame in combined_output_frame.groupby("component_name"):
    fig, axes = plt.subplots(1, 1, figsize=(48 / 5, 32 / 5))
    # A separate case is needed for the pipe diameters
    plt.axvline(x=0, linestyle="--", color="grey", linewidth=1)
    if component_name == "Pipe":
        # Create a new frame containing the inner diameter and outer diameter values
        pipe_frame = sub_frame.copy()
        pipe_frame["outer_diameter"] = pipe_frame["parameter_value"].astype(float)
        pipe_frame["inner_diameter"] = None
        for index, row in pipe_frame.iterrows():
            pipe_frame.at[index, "inner_diameter"] = float(
                pipe_diameter_regex.match(row["parameter_name"]).group("inner_diameter")
            )
        pipe_frame["thickness"] = (
            pipe_frame["outer_diameter"] - pipe_frame["inner_diameter"]
        ).round(2)
        axis = axes
        sns.scatterplot(
            pipe_frame,
            x="squared_thermal_efficiency_change",
            y="outer_diameter",
            hue="thickness",
            marker="D",
            s=50,
            linewidth=0,
            alpha=0.2,
            palette=(
                this_palette := sns.cubehelix_palette(start=0, rot=-0.2, n_colors=14)
            ),
        )
        norm = plt.Normalize(
            pipe_frame["squared_thermal_efficiency_change"].min(),
            pipe_frame["squared_thermal_efficiency_change"].max(),
        )
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "Custom", this_palette.as_hex(), len(set(pipe_frame["inner_diameter"]))
            ),
            norm=norm,
        )
        colorbar = axis.figure.colorbar(
            scalar_mappable, ax=axis, label="Inner pipe diameter / mm"
        )
        axis.get_legend().remove()
        plt.xlabel("Squared fractional change in thermal efficiency")
        plt.ylabel("Outer pipe diameter / mm")
    else:
        order_by_square = (
            sub_frame.groupby(by=["parameter_name"])[
                "squared_thermal_efficiency_change"
            ]
            .mean()
            .sort_values()
            .iloc[::-1]
            .index
        )
        if (num_parameter_names := len(set(sub_frame["parameter_name"]))) > len(
            sns.color_palette()
        ):
            norm = plt.Normalize(0, num_parameter_names)
            scalar_mappable = plt.cm.ScalarMappable(
                cmap=mcolors.LinearSegmentedColormap.from_list(
                    "Custom", sns.color_palette().as_hex(), num_parameter_names
                ),
                norm=norm,
            )
        sns.violinplot(
            sub_frame,
            x="squared_thermal_efficiency_change",
            y="parameter_name",
            ax=(axis := axes),
            cut=0,
            inner=None,
            linewidth=1,
            order=order_by_square,
            hue_order=order_by_square,
            palette=blue_first_thesis_palette,
            hue="parameter_name",
            alpha=0.8,
        )
        sns.stripplot(
            sub_frame,
            x="squared_thermal_efficiency_change",
            y="parameter_name",
            palette=blue_first_thesis_palette,
            hue_order=order_by_square,
            ax=axis,
            alpha=0.8,
            linewidth=0,
            marker="D",
            size=1,
            dodge=False,
            order=order_by_square,
        )
        plt.xlabel("Squared fractional change in thermal efficiency")
        plt.ylabel("Parameter name")
    # Remove top and right axes, keep bottom and left with ticks
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(bottom=True, left=True)  # Enable ticks on bottom and left axes
    # Adjust margins (optional, adjust values for desired spacing)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.xlim(round(10 * vmin, -1) / 10, round(10 * vmax, 1) / 10)
    plt.title(
        (
            "PV-T"
            if component_name == "Pvt"
            else "PV" if component_name == "Pv" else component_name.capitalize()
        ),
        weight="bold",
    )
    plt.savefig(
        f"squared_fractions_violins_for_{component_name}_with_border_{INDEX}.png",
        transparent=True,
        dpi=400,
        bbox_inches="tight",
    )

# Determine the range to used based on the extreme range
vmax = combined_output_frame["percentage_thermal_efficiency_change"].dropna().max()
vmin = combined_output_frame["percentage_thermal_efficiency_change"].dropna().min()

sns.set_palette(blue_first_thesis_palette)
sns.set_style("ticks")

for component_name, sub_frame in combined_output_frame.groupby("component_name"):
    fig, axes = plt.subplots(1, 1, figsize=(48 / 5, 32 / 5))
    # A separate case is needed for the pipe diameters
    plt.axvline(x=0, linestyle="--", color="grey", linewidth=1)
    if component_name == "Pipe":
        # Create a new frame containing the inner diameter and outer diameter values
        pipe_frame = sub_frame.copy()
        pipe_frame["outer_diameter"] = pipe_frame["parameter_value"].astype(float)
        pipe_frame["inner_diameter"] = None
        for index, row in pipe_frame.iterrows():
            pipe_frame.at[index, "inner_diameter"] = float(
                pipe_diameter_regex.match(row["parameter_name"]).group("inner_diameter")
            )
        pipe_frame["thickness"] = (
            pipe_frame["outer_diameter"] - pipe_frame["inner_diameter"]
        ).round(2)
        axis = axes
        sns.scatterplot(
            pipe_frame,
            x="percentage_thermal_efficiency_change",
            y="outer_diameter",
            hue="thickness",
            marker="D",
            s=50,
            linewidth=0,
            alpha=0.2,
            palette=(
                this_palette := sns.cubehelix_palette(start=0, rot=-0.2, n_colors=14)
            ),
        )
        norm = plt.Normalize(
            pipe_frame["percentage_thermal_efficiency_change"].min(),
            pipe_frame["percentage_thermal_efficiency_change"].max(),
        )
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "Custom", this_palette.as_hex(), len(set(pipe_frame["inner_diameter"]))
            ),
            norm=norm,
        )
        colorbar = axis.figure.colorbar(
            scalar_mappable, ax=axis, label="Inner pipe diameter / mm"
        )
        axis.get_legend().remove()
        plt.xlabel("Percentage change in thermal efficiency")
        plt.ylabel("Outer pipe diameter")
    else:
        order_by_square = (
            sub_frame.groupby(by=["parameter_name"])[
                "squared_thermal_efficiency_change"
            ]
            .mean()
            .sort_values()
            .iloc[::-1]
            .index
        )
        if (num_parameter_names := len(set(sub_frame["parameter_name"]))) > len(
            sns.color_palette()
        ):
            norm = plt.Normalize(0, num_parameter_names)
            scalar_mappable = plt.cm.ScalarMappable(
                cmap=mcolors.LinearSegmentedColormap.from_list(
                    "Custom", sns.color_palette().as_hex(), num_parameter_names
                ),
                norm=norm,
            )
        sns.violinplot(
            sub_frame,
            x="percentage_thermal_efficiency_change",
            y="parameter_name",
            ax=(axis := axes),
            cut=0,
            inner=None,
            linewidth=1,
            order=order_by_square,
            hue_order=order_by_square,
            palette=blue_first_thesis_palette,
            hue="parameter_name",
            alpha=0.8,
        )
        sns.stripplot(
            sub_frame,
            x="percentage_thermal_efficiency_change",
            y="parameter_name",
            palette=blue_first_thesis_palette,
            hue_order=order_by_square,
            ax=axis,
            alpha=0.8,
            linewidth=0,
            marker="D",
            size=1,
            dodge=False,
            order=order_by_square,
        )
        plt.xlabel("Percentage change in thermal efficiency")
        plt.ylabel("Parameter name")
    # Remove top and right axes, keep bottom and left with ticks
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(bottom=True, left=True)  # Enable ticks on bottom and left axes
    # Adjust margins (optional, adjust values for desired spacing)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.xlim(round(10 * vmin, -1) / 10, round(10 * vmax, 1) / 10)
    plt.title(
        (
            "PV-T"
            if component_name == "Pvt"
            else "PV" if component_name == "Pv" else component_name.capitalize()
        ),
        weight="bold",
    )
    plt.savefig(
        f"percentage_fractions_violins_for_{component_name}_with_border_{INDEX}.png",
        transparent=True,
        dpi=400,
        bbox_inches="tight",
    )

plt.close()

# Plot the fractional, squared, and percentage changes but without limits
# Determine the range to used based on the extreme range
sns.set_palette(blue_first_thesis_palette)
sns.set_style("ticks")

for component_name, sub_frame in combined_output_frame.groupby("component_name"):
    fig, axes = plt.subplots(1, 1, figsize=(48 / 5, 32 / 5))
    # A separate case is needed for the pipe diameters
    plt.axvline(x=0, linestyle="--", color="grey", linewidth=1)
    if component_name == "Pipe":
        # Create a new frame containing the inner diameter and outer diameter values
        pipe_frame = sub_frame.copy()
        pipe_frame["outer_diameter"] = pipe_frame["parameter_value"].astype(float)
        pipe_frame["inner_diameter"] = None
        for index, row in pipe_frame.iterrows():
            pipe_frame.at[index, "inner_diameter"] = float(
                pipe_diameter_regex.match(row["parameter_name"]).group("inner_diameter")
            )
        pipe_frame["thickness"] = (
            pipe_frame["outer_diameter"] - pipe_frame["inner_diameter"]
        ).round(2)
        axis = axes
        sns.scatterplot(
            pipe_frame,
            x="fractional_thermal_efficiency_change",
            y="outer_diameter",
            hue="thickness",
            marker="D",
            s=50,
            linewidth=0,
            alpha=0.2,
            palette=(
                this_palette := sns.cubehelix_palette(start=0, rot=-0.2, n_colors=14)
            ),
        )
        norm = plt.Normalize(
            pipe_frame["fractional_thermal_efficiency_change"].min(),
            pipe_frame["fractional_thermal_efficiency_change"].max(),
        )
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "Custom", this_palette.as_hex(), len(set(pipe_frame["inner_diameter"]))
            ),
            norm=norm,
        )
        colorbar = axis.figure.colorbar(
            scalar_mappable, ax=axis, label="Inner pipe diameter / mm"
        )
        axis.get_legend().remove()
        plt.xlabel("Fractional change in thermal efficiency")
        plt.ylabel("Outer pipe diameter / mm")
    else:
        order_by_square = (
            sub_frame.groupby(by=["parameter_name"])[
                "squared_thermal_efficiency_change"
            ]
            .mean()
            .sort_values()
            .iloc[::-1]
            .index
        )
        if (num_parameter_names := len(set(sub_frame["parameter_name"]))) > len(
            sns.color_palette()
        ):
            norm = plt.Normalize(0, num_parameter_names)
            scalar_mappable = plt.cm.ScalarMappable(
                cmap=mcolors.LinearSegmentedColormap.from_list(
                    "Custom", sns.color_palette().as_hex(), num_parameter_names
                ),
                norm=norm,
            )
        sns.violinplot(
            sub_frame,
            x="fractional_thermal_efficiency_change",
            y="parameter_name",
            ax=(axis := axes),
            cut=0,
            inner=None,
            linewidth=1,
            order=order_by_square,
            hue_order=order_by_square,
            palette=blue_first_thesis_palette,
            hue="parameter_name",
            alpha=0.8,
        )
        sns.stripplot(
            sub_frame,
            x="fractional_thermal_efficiency_change",
            y="parameter_name",
            palette=blue_first_thesis_palette,
            hue_order=order_by_square,
            ax=axis,
            alpha=0.8,
            linewidth=0,
            marker="D",
            size=1,
            dodge=False,
            order=order_by_square,
        )
        plt.xlabel("Fractional change in thermal efficiency")
        plt.ylabel("Parameter name")
    # Remove top and right axes, keep bottom and left with ticks
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(bottom=True, left=True)  # Enable ticks on bottom and left axes
    # Adjust margins (optional, adjust values for desired spacing)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.title(
        (
            "PV-T"
            if component_name == "Pvt"
            else "PV" if component_name == "Pv" else component_name.capitalize()
        ),
        weight="bold",
    )
    plt.savefig(
        f"no_lim_fractions_violins_for_{component_name}_with_border_{INDEX}.png",
        transparent=True,
        dpi=400,
        bbox_inches="tight",
    )

plt.close()

# The squared changes, but without limits imposed
sns.set_palette(blue_first_thesis_palette)
sns.set_style("ticks")

for component_name, sub_frame in combined_output_frame.groupby("component_name"):
    fig, axes = plt.subplots(1, 1, figsize=(48 / 5, 32 / 5))
    # A separate case is needed for the pipe diameters
    plt.axvline(x=0, linestyle="--", color="grey", linewidth=1)
    if component_name == "Pipe":
        # Create a new frame containing the inner diameter and outer diameter values
        pipe_frame = sub_frame.copy()
        pipe_frame["outer_diameter"] = pipe_frame["parameter_value"].astype(float)
        pipe_frame["inner_diameter"] = None
        for index, row in pipe_frame.iterrows():
            pipe_frame.at[index, "inner_diameter"] = float(
                pipe_diameter_regex.match(row["parameter_name"]).group("inner_diameter")
            )
        pipe_frame["thickness"] = (
            pipe_frame["outer_diameter"] - pipe_frame["inner_diameter"]
        ).round(2)
        axis = axes
        sns.scatterplot(
            pipe_frame,
            x="squared_thermal_efficiency_change",
            y="outer_diameter",
            hue="thickness",
            marker="D",
            s=50,
            linewidth=0,
            alpha=0.2,
            palette=(
                this_palette := sns.cubehelix_palette(start=0, rot=-0.2, n_colors=14)
            ),
        )
        norm = plt.Normalize(
            pipe_frame["squared_thermal_efficiency_change"].min(),
            pipe_frame["squared_thermal_efficiency_change"].max(),
        )
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "Custom", this_palette.as_hex(), len(set(pipe_frame["inner_diameter"]))
            ),
            norm=norm,
        )
        colorbar = axis.figure.colorbar(
            scalar_mappable, ax=axis, label="Inner pipe diameter / mm"
        )
        axis.get_legend().remove()
        plt.xlabel("Squared fractional change in thermal efficiency")
        plt.ylabel("Outer pipe diameter / mm")
    else:
        order_by_square = (
            sub_frame.groupby(by=["parameter_name"])[
                "squared_thermal_efficiency_change"
            ]
            .mean()
            .sort_values()
            .iloc[::-1]
            .index
        )
        if (num_parameter_names := len(set(sub_frame["parameter_name"]))) > len(
            sns.color_palette()
        ):
            norm = plt.Normalize(0, num_parameter_names)
            scalar_mappable = plt.cm.ScalarMappable(
                cmap=mcolors.LinearSegmentedColormap.from_list(
                    "Custom", sns.color_palette().as_hex(), num_parameter_names
                ),
                norm=norm,
            )
        sns.violinplot(
            sub_frame,
            x="squared_thermal_efficiency_change",
            y="parameter_name",
            ax=(axis := axes),
            cut=0,
            inner=None,
            linewidth=1,
            order=order_by_square,
            hue_order=order_by_square,
            palette=blue_first_thesis_palette,
            hue="parameter_name",
            alpha=0.8,
        )
        sns.stripplot(
            sub_frame,
            x="squared_thermal_efficiency_change",
            y="parameter_name",
            palette=blue_first_thesis_palette,
            hue_order=order_by_square,
            ax=axis,
            alpha=0.8,
            linewidth=0,
            marker="D",
            size=1,
            dodge=False,
            order=order_by_square,
        )
        plt.xlabel("Squared fractional change in thermal efficiency")
        plt.ylabel("Parameter name")
    # Remove top and right axes, keep bottom and left with ticks
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(bottom=True, left=True)  # Enable ticks on bottom and left axes
    # Adjust margins (optional, adjust values for desired spacing)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.title(
        (
            "PV-T"
            if component_name == "Pvt"
            else "PV" if component_name == "Pv" else component_name.capitalize()
        ),
        weight="bold",
    )
    plt.savefig(
        f"no_lim_squared_fractions_violins_for_{component_name}_with_border_{INDEX}.png",
        transparent=True,
        dpi=400,
        bbox_inches="tight",
    )

# Percentage changes, without limits imposed.
sns.set_palette(blue_first_thesis_palette)
sns.set_style("ticks")

for component_name, sub_frame in combined_output_frame.groupby("component_name"):
    fig, axes = plt.subplots(1, 1, figsize=(48 / 5, 32 / 5))
    # A separate case is needed for the pipe diameters
    plt.axvline(x=0, linestyle="--", color="grey", linewidth=1)
    if component_name == "Pipe":
        # Create a new frame containing the inner diameter and outer diameter values
        pipe_frame = sub_frame.copy()
        pipe_frame["outer_diameter"] = pipe_frame["parameter_value"].astype(float)
        pipe_frame["inner_diameter"] = None
        for index, row in pipe_frame.iterrows():
            pipe_frame.at[index, "inner_diameter"] = float(
                pipe_diameter_regex.match(row["parameter_name"]).group("inner_diameter")
            )
        pipe_frame["thickness"] = (
            pipe_frame["outer_diameter"] - pipe_frame["inner_diameter"]
        ).round(2)
        axis = axes
        sns.scatterplot(
            pipe_frame,
            x="percentage_thermal_efficiency_change",
            y="outer_diameter",
            hue="thickness",
            marker="D",
            s=50,
            linewidth=0,
            alpha=0.2,
            palette=(
                this_palette := sns.cubehelix_palette(start=0, rot=-0.2, n_colors=14)
            ),
        )
        norm = plt.Normalize(
            pipe_frame["percentage_thermal_efficiency_change"].min(),
            pipe_frame["percentage_thermal_efficiency_change"].max(),
        )
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "Custom", this_palette.as_hex(), len(set(pipe_frame["inner_diameter"]))
            ),
            norm=norm,
        )
        colorbar = axis.figure.colorbar(
            scalar_mappable, ax=axis, label="Inner pipe diameter / mm"
        )
        axis.get_legend().remove()
        plt.xlabel("Percentage change in thermal efficiency")
        plt.ylabel("Outer pipe diameter / mm")
    else:
        order_by_square = (
            sub_frame.groupby(by=["parameter_name"])[
                "squared_thermal_efficiency_change"
            ]
            .mean()
            .sort_values()
            .iloc[::-1]
            .index
        )
        if (num_parameter_names := len(set(sub_frame["parameter_name"]))) > len(
            sns.color_palette()
        ):
            norm = plt.Normalize(0, num_parameter_names)
            scalar_mappable = plt.cm.ScalarMappable(
                cmap=mcolors.LinearSegmentedColormap.from_list(
                    "Custom", sns.color_palette().as_hex(), num_parameter_names
                ),
                norm=norm,
            )
        sns.violinplot(
            sub_frame,
            x="percentage_thermal_efficiency_change",
            y="parameter_name",
            ax=(axis := axes),
            cut=0,
            inner=None,
            linewidth=1,
            order=order_by_square,
            hue_order=order_by_square,
            palette=blue_first_thesis_palette,
            hue="parameter_name",
            alpha=0.8,
        )
        sns.stripplot(
            sub_frame,
            x="percentage_thermal_efficiency_change",
            y="parameter_name",
            palette=blue_first_thesis_palette,
            hue_order=order_by_square,
            ax=axis,
            alpha=0.8,
            linewidth=0,
            marker="D",
            size=1,
            dodge=False,
            order=order_by_square,
        )
        plt.xlabel("Percentage change in thermal efficiency")
        plt.ylabel("Parameter name")
    # Remove top and right axes, keep bottom and left with ticks
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(bottom=True, left=True)  # Enable ticks on bottom and left axes
    # Adjust margins (optional, adjust values for desired spacing)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.title(
        (
            "PV-T"
            if component_name == "Pvt"
            else "PV" if component_name == "Pv" else component_name.capitalize()
        ),
        weight="bold",
    )
    plt.savefig(
        f"no_lim_percentage_fractions_violins_for_{component_name}_with_border_{INDEX}.png",
        transparent=True,
        dpi=400,
        bbox_inches="tight",
    )

plt.show()

plt.close()
