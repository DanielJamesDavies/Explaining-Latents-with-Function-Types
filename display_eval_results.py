import matplotlib.pyplot as plt
import pandas as pd
import json
import mplcyberpunk
import numpy as np

plt.style.use("cyberpunk")
background_colour = "#080808"
plt.rcParams['axes.facecolor'] = background_colour
plt.rcParams['grid.color'] = background_colour


# Load Data
eval_data = []
for layer_index in range(12):
    eval_data.append([])
    with open(f'./latent_data/{layer_index}/eval_results.jsonl', 'r') as f:
        for i, line in enumerate(f):
            eval_data[layer_index].append(json.loads(line))





# Fractions of Successful Explanations over Layers

eval_layers_success_fractions = [
    len([item for item in layer_eval_data if "success" in item and item["success"] is True]) / len([item for item in layer_eval_data if "success" in item])
    for layer_eval_data in eval_data
]

layer_labels = [str(i+1) for i in range(12)]

layer_success_df = pd.DataFrame(
   dict(
      layers=layer_labels,
      values=eval_layers_success_fractions
   )
)

plt.figure(figsize=(1920 / 160, 1080 / 160), facecolor=background_colour)
plt.plot(layer_labels, eval_layers_success_fractions, color="#0044ff")
for i, value in enumerate(list(layer_success_df["values"])):
    plt.text(i, value+0.01, "{:.2f}".format(value), ha='center', va="bottom", fontsize=13)
plt.xticks(ha='center', fontsize=13)
plt.xlabel("Layer", fontsize=16, labelpad=20)
plt.ylabel("Fraction Successful", fontsize=16, labelpad=20)
plt.title("Fraction of Successful Explanations over Layers", fontsize=18, pad=40)
mplcyberpunk.add_glow_effects(gradient_fill=True)
plt.subplots_adjust(left=0.1, bottom=0.14, right=0.97, top=0.85, wspace=0.2, hspace=0.2)
plt.savefig("Fractions of Successful Explanations over Layers.png", facecolor=background_colour, dpi=300)





# Success of Function Types over Layers

function_type_names = ["SpecificToken()", "ConnectingTokens()", "DetectingDatasetTopic()"]
data = {
    "benchmarks": [str(i+1) for j in range(3) for i in range(12)],
    "subcategories": [function_type_names[j] for j in range(3) for i in range(12)],
    "values": [
        len([item for item in layer_eval_data if "success" in item and item["success"] is True and item["type"] == function_type]) / len([item for item in layer_eval_data if "success" in item])
            for function_type in ["top-token", "connecting-tokens", "detecting-dataset-topic"]
            for layer_eval_data in eval_data
        ]
}
df = pd.DataFrame(data)
df['benchmarks'] = pd.to_numeric(df['benchmarks'])

pivoted_data = df.pivot(index="benchmarks", columns="subcategories", values="values")
pivoted_data = pivoted_data[function_type_names]

x = np.arange(len(pivoted_data.index))
width = 0.16

fig, ax = plt.subplots(figsize=(1920 / 160, 1080 / 160), facecolor=background_colour)
colors = ["#0044ff", "#ff4400", "#00ff44"]
for i, subcategory in enumerate(pivoted_data.columns):
    bars = ax.bar(x + i * (width + 0.05), pivoted_data[subcategory], width, label=subcategory, color=colors[i])
    mplcyberpunk.add_bar_gradient(bars=bars)

ax.set_xlabel("Function Types over Layers", fontsize=16, labelpad=20)
ax.set_ylabel("Fraction Successful", fontsize=16, labelpad=20)
ax.set_title("Success of Function Types over Layers", fontsize=18, pad=40)
ax.set_xticks(x + width / 2 + 0.1)
ax.set_xticklabels(pivoted_data.index)
ax.legend(title="Function Types")

for i, bar_group in enumerate(ax.containers):
    for j, bar in enumerate(bar_group):
        height = bar.get_height()
        other_bar_height = ax.containers[1 if i == 0 else 0][j].get_height()
        padding = 2 if height > other_bar_height else 1
        ax.text(
            bar.get_x() + bar.get_width() / 2 + (-0.1 if i == 0 else 0.1 if height == other_bar_height else 0),
            height + padding / 100,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

plt.subplots_adjust(left=0.1, bottom=0.14, right=0.97, top=0.85, wspace=0.2, hspace=0.2)
plt.savefig("Success of Function Types over Layers.png", facecolor=background_colour, dpi=300)





# Success of SpecificToken() by Model Type

data = {
    "benchmarks": [str(i+1) for j in range(2) for i in range(12)],
    "subcategories": [["Variant", "Non-Variant"][j] for j in range(2) for i in range(12)],
    "values": [
        len([item for item in layer_eval_data if "success" in item and item["success"] is True and item["type"] == "top-token" and item["is_variant"] is is_variant]) / len([item for item in layer_eval_data if "success" in item and item["type"] == "top-token" and "is_variant" in item])
            for is_variant in [True, False]
            for layer_eval_data in eval_data
        ]
}
df = pd.DataFrame(data)
df['benchmarks'] = pd.to_numeric(df['benchmarks'])

pivoted_data = df.pivot(index="benchmarks", columns="subcategories", values="values")

x = np.arange(len(pivoted_data.index))
width = 0.16

fig, ax = plt.subplots(figsize=(1920 / 160, 1080 / 160), facecolor=background_colour)
colors = ["#0044ff", "#ff4400", "#00ff44"]
for i, subcategory in enumerate(pivoted_data.columns):
    bars = ax.bar(x + i * (width + 0.05), pivoted_data[subcategory], width, label=subcategory, color=colors[i])
    mplcyberpunk.add_bar_gradient(bars=bars)

ax.set_xlabel("Function Types over Layers", fontsize=16, labelpad=20)
ax.set_ylabel("Fraction Successful", fontsize=16, labelpad=20)
ax.set_title("Success of SpecificToken() by Model Type", fontsize=18, pad=40)
ax.set_xticks(x + width / 2 + 0.1)
ax.set_xticklabels(pivoted_data.index)
ax.legend(title="Function Types")

for i, bar_group in enumerate(ax.containers):
    for j, bar in enumerate(bar_group):
        height = bar.get_height()
        other_bar_height = ax.containers[1 if i == 0 else 0][j].get_height()
        padding = 2 if height > other_bar_height else 1
        ax.text(
            bar.get_x() + bar.get_width() / 2 + (-0.1 if i == 0 else 0.1 if height == other_bar_height else 0),
            height + padding / 100,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

plt.subplots_adjust(left=0.1, bottom=0.14, right=0.97, top=0.85, wspace=0.2, hspace=0.2)
plt.savefig("Success of SpecificToken() by Model Type.png", facecolor=background_colour, dpi=300)
