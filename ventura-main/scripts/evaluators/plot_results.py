import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# ---- Data ----
data = {
    "Model": ["Ventura", "Ventura-P", "LeLaN", "Convoi"],
    "Obstacle Avoidance Seen": [13/15, 10/15, 9/15, 8/15],
    "Obstacle Avoidance Unseen": [4/5, 1/5, 1/5, 3/5],
    "Object Goal Reaching Seen": [9/10, 5/10, 8/10, 7/10],
    "Object Goal Reaching Unseen": [7/10, 4/10, 3/10, 7/10],
    "Terrain Aware Seen": [6/6, 4/6, 3/6, 4/6],
    "Terrain Aware Unseen": [5/6, 2/6, 2/6, 3/6],
}
df = pd.DataFrame(data)
df_long = df.melt(id_vars="Model", var_name="Category", value_name="SuccessRate")
df_long[["Task", "Env"]] = df_long["Category"].str.rsplit(" ", n=1, expand=True)
df_long.drop(columns=["Category"], inplace=True)

# ---- Plot ----
sns.set(style="whitegrid", context="talk")
tasks = df_long["Task"].unique()
models = df_long["Model"].unique()
palette = sns.color_palette("Set2", len(models))
color_map = dict(zip(models, palette))

fig, axes = plt.subplots(1, len(tasks), figsize=(5*len(tasks)+2, 6), sharey=True)

for i, task in enumerate(tasks):
    ax = axes[i]
    subset = df_long[df_long["Task"] == task]
    sns.barplot(
        data=subset,
        x="Env",
        y="SuccessRate",
        hue="Model",
        hue_order=models,
        palette=palette,
        ax=ax,
        legend=False,
    )
    ax.set_title(task, fontsize=24, fontweight="bold")
    ax.set_ylabel("Success Rate" if i == 0 else "", fontsize=20, fontweight="bold")
    ax.set_xlabel("Environment", fontsize=20, fontweight="bold")
    ax.set_ylim(0, 1.05)

    # Minor ticks + dotted grids
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

# ---- Shared legend to the right of last subplot ----
handles = [Patch(facecolor=color_map[m], label=m) for m in models]
fig.legend(
    handles,
    [m for m in models],
    loc="center left",
    bbox_to_anchor=(0.83, 0.78),
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space on the right for legend
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")