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

eval_layers_success_fractions = [len([item for item in layer_eval_data if "success" in item and item["success"] is True]) / len([item for item in layer_eval_data if "success" in item]) for layer_eval_data in eval_data]

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
plt.savefig("Fraction of Successful Explanations over Layers.png", facecolor=background_colour, dpi=300)




eval_layers_success_fractions = [len([item for item in layer_eval_data if "success" in item and item["success"] is True]) / len([item for item in layer_eval_data if "success" in item]) for layer_eval_data in eval_data]

data = {
    "benchmarks": [str(i+1) for j in range(3) for i in range(12)],
    "subcategories": [["SpecificToken()", "ConnectingTokens()", "DetectingDatasetTopic()"][j] for j in range(3) for i in range(12)],
    "values": [(i+1) for j in range(3) for i in range(12)]
}
df = pd.DataFrame(data)

# Pivot data to prepare for grouped bar chart
pivoted_data = df.pivot(index="benchmarks", columns="subcategories", values="values")

# Group settings
x = np.arange(len(pivoted_data.index))
width = 0.16

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
for i, subcategory in enumerate(pivoted_data.columns):
    print(pivoted_data[subcategory])
    bars = ax.bar(x + i * width, pivoted_data[subcategory], width, label=subcategory, color="#0044ff")
    mplcyberpunk.add_bar_gradient(bars=bars)

# Add labels and title
ax.set_xlabel("Benchmarks")
ax.set_ylabel("Scores")
ax.set_title("Average Scores by Benchmark and Group")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(pivoted_data.index)
ax.legend(title="Groups")

# Show values on top of bars
for bar_group in ax.containers:
    ax.bar_label(bar_group, fmt="%.2f", padding=3, fontsize=10)

plt.subplots_adjust(left=0.1, bottom=0.14, right=0.97, top=0.85, wspace=0.2, hspace=0.2)
plt.savefig("B.png", facecolor=background_colour, dpi=300)
