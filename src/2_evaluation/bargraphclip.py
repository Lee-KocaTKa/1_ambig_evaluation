import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------- CLEAN STYLE ----------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

models = ["CLIP-ViT", "CLIP-RN50", "CLIP-RN101" ,"SigLIP", "SigLIP2", "MetaCLIP", "MetaCLIP2", "OpenCLIP-ViT","OpenCLIP-convnext", "EVA-CLIP", "Human"]
models = ["MetaCLIP2","Human"]
types = ["VP", "PP", "Anaph", "Ellip", "Adj", "Vb", "Conj", "All"]

# results[type][model] = (I2T, T2I, Dual)
results = {
    "VP": {
        #"CLIP-ViT": (0.50, 0.515, 0.25),
        #"CLIP-RN50": (0.52, 0.485, 0.275),
        #"CLIP-RN101": (0.475, 0.505, 0.245),
        #"SigLIP": (0.54, 0.495, 0.31),
        #"SigLIP2": (0.525, 0.51, 0.265),
        #"MetaCLIP": (0.53, 0.49, 0.265),
        "MetaCLIP2": (0.51, 0.495, 0.275),
        #"OpenCLIP-ViT": (0.485, 0.495, 0.230),
        #"OpenCLIP-convnext": (0.54, 0.505, 0.3),
        #"EVA-CLIP": (0.49, 0.5, 0.28),
        "Human": (0.965, 0.985, 0.960),
    },
    "PP": {
        #"CLIP-ViT": (0.555, 0.50, 0.32),
        #"CLIP-RN50": (0.555, 0.51, 0.295),
        #"CLIP-RN101": (0.45, 0.495, 0.23),
        #"SigLIP": (0.570, 0.5, 0.265),
        #"SigLIP2": (0.525, 0.535, 0.3),
        #"MetaCLIP": (0.56, 0.51, 0.305),
        "MetaCLIP2": (0.545, 0.500, 0.265),
        #"OpenCLIP-ViT": (0.53, 0.52, 0.26),
        #"OpenCLIP-convnext": (0.515, 0.51, 0.265),
        #"EVA-CLIP": (0.545, 0.495, 0.275),
        "Human": (0.925, 0.970, 0.915),
    },
    "Anaph": {
        #"CLIP-ViT": (0.537, 0.498, 0.269),
        #"CLIP-RN50": (0.552, 0.517, 0.309),
        #"CLIP-RN101": (0.527, 0.488, 0.279),
        #"SigLIP": (0.512, 0.503, 0.279),
        #"SigLIP2": (0.503, 0.488, 0.249),
        #"MetaCLIP": (0.542, 0.488, 0.259),
        "MetaCLIP2": (0.532, 0.503, 0.294),
        #"OpenCLIP-ViT": (0.567, 0.488, 0.284),
        #"OpenCLIP-convnext": (0.522, 0.508, 0.299),
        #"EVA-CLIP": (0.517, 0.508, 0.313),
        "Human": (0.945, 0.881, 0.851),
    },
    "Ellip": {
        #"CLIP-ViT": (0.545, 0.49, 0.307),
        #"CLIP-RN50": (0.525, 0.53, 0.307),
        #"CLIP-RN101": (0.525, 0.5, 0.277),
        #"SigLIP": (0.5, 0.505, 0.223),
        #"SigLIP2": (0.495, 0.5, 0.287),
        #"MetaCLIP": (0.559, 0.564, 0.302),
        "MetaCLIP2": (0.559, 0.52, 0.307),
        #"OpenCLIP-ViT": (0.604, 0.5, 0.312),
        #"OpenCLIP-convnext": (0.555, 0.535, 0.337),
        #"EVA-CLIP": (0.579, 0.52, 0.297),
        "Human": (0.896, 0.926, 0.847),
    },
    "Adj": {
        #"CLIP-ViT": (0.535, 0.52, 0.31),
        #"CLIP-RN50": (0.55, 0.5, 0.295),
        #"CLIP-RN101": (0.585, 0.505, 0.295),
        #"SigLIP": (0.505, 0.505, 0.215),
        #"SigLIP2": (0.495, 0.52, 0.295),
        #"MetaCLIP": (0.565, 0.51, 0.330),
        "MetaCLIP2": (0.585, 0.515, 0.315),
        #"OpenCLIP-ViT": (0.53, 0.525, 0.255),
        #"OpenCLIP-convnext": (0.615, 0.515, 0.305),
        #"EVA-CLIP": (0.585, 0.525, 0.31),
        "Human": (0.895, 0.905, 0.855),
    },
    "Vb": {
        #"CLIP-ViT": (0.59, 0.53, 0.305),
        #"CLIP-RN50": (0.555, 0.51, 0.26),
        #"CLIP-RN101": (0.58, 0.51, 0.315),
        #"SigLIP": (0.555, 0.505, 0.330),
        #"SigLIP2": (0.495, 0.49, 0.28),
        #"MetaCLIP": (0.55, 0.525, 0.315),
        "MetaCLIP2": (0.55, 0.515, 0.330),
        #"OpenCLIP-ViT": (0.6, 0.515, 0.335),
        #"OpenCLIP-convnext": (0.6, 0.52, 0.31),
        #"EVA-CLIP": (0.605, 0.510, 0.365),
        "Human": (0.945, 0.920, 0.915),
    },
    "Conj": {
        #"CLIP-ViT": (0.503, 0.37, 0.170),
        #"CLIP-RN50": (0.507, 0.36, 0.197),
        #"CLIP-RN101": (0.54, 0.353, 0.187),
        #"SigLIP": (0.46, 0.383, 0.203),
        #"SigLIP2": (0.35, 0.327, 0.110),
        #"MetaCLIP": (0.517, 0.363, 0.197),
        "MetaCLIP2": (0.59, 0.387, 0.22),
        #"OpenCLIP-ViT": (0.54, 0.377, 0.197),
        #"OpenCLIP-convnext": (0.557, 0.37, 0.217),
        #"EVA-CLIP": (0.54, 0.387, 0.203),
        "Human": (0.933, 0.93, 0.913),
    },
    "All": {
        #"CLIP-ViT": (0.536, 0.481, 0.269),
        #"CLIP-RN50": (0.536, 0.479, 0.272),
        #"CLIP-RN101": (0.527, 0.471, 0.256),
        #"SigLIP": (0.516, 0.478, 0.257),
        #"SigLIP2": (0.475, 0.471, 0.246),
        #"MetaCLIP": (0.544, 0.484, 0.276),
        "MetaCLIP2": (0.556, 0.484, 0.282),
        #"OpenCLIP-ViT": (0.55, 0.481, 0.263),
        #"OpenCLIP-convnext": (0.558, 0.486, 0.285),
        #"EVA-CLIP": (0.551, 0.485, 0.286),
        "Human": (0.929, 0.931, 0.896),
    },
}

# ---------------- GRID PLOT ----------------

fig, axes = plt.subplots(
    nrows=len(models), 
    ncols=len(types),
    figsize=(28, 6),     # ★ 줄이고 싶으면 숫자를 더 줄이면 됨
    gridspec_kw={"height_ratios": [0.4]*len(models)}  # ★ subplot height 줄임
)

bar_width = 0.6
x = np.arange(3)

for i, model in enumerate(models):
    for j, amb_type in enumerate(types):
        ax = axes[i, j]

        vals = np.array(results[amb_type][model]) * 100  # NORMAL scale, no shrinking
        ax.bar(x, vals, color=["#4C72B0", "#DD8452", "#55A868"], width=bar_width)

        # Title only on top row
        if i == 0:
            ax.set_title(amb_type, fontsize=12, fontweight="bold")

        # Model labels on leftmost column
        if j == 0:
            ax.set_ylabel(
                model, fontsize=12, fontweight="bold",
                rotation=0, labelpad=25, va="center"
            )

        # Keep true axis range
        ax.set_ylim(0, 100)

        # Clean axes
        ax.set_xticks([])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis='y', labelsize=8)

        # Baselines
        if amb_type == "Conj":
            ax.axhline(33, color='red', linestyle='-', linewidth=1)
            ax.axhline(11, color='red', linestyle='--', linewidth=1)
        else:
            ax.axhline(50, color='red', linestyle='-', linewidth=1)
            ax.axhline(25, color='red', linestyle='--', linewidth=1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

# Global legend
fig.legend(
    ["Random Chance (except Dual)", "Random Chance (Dual)", "T2I", "I2T", "Dual"],
    loc='upper center', ncol=5, fontsize=10, frameon=False
)

# Compact layout
plt.tight_layout(rect=[0, 0, 1, 0.93], pad=0.5, w_pad=0.8, h_pad=0.5)

plt.savefig("classification_grid.png", dpi=300, bbox_inches="tight")
plt.show()
