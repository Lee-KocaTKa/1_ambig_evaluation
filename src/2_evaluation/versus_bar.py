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
models = ["MetaCLIP2"]
types = ["VP", "PP", "Anaph", "Ellip", "Adj", "Vb", "Conj", "All"]

# results[type][model] = (I2T, T2I, Dual)
results = {
    "VP": {
        #"CLIP-ViT": (0.71, 0.17),
        #"CLIP-RN50": (0.52, 0.485, 0.275),
        #"CLIP-RN101": (0.475, 0.505, 0.245),
        #"SigLIP": (0.84, 0.055),
        #"SigLIP2": (0.545, 0.67),
        #"MetaCLIP": (0.82, 0.145),
        "MetaCLIP2": (0.765, 0.065),
        #"OpenCLIP-ViT": (0.665, 0.17),
        #"OpenCLIP-convnext": (0.54, 0.505, 0.3),
        #"EVA-CLIP": (0.49, 0.5, 0.28),
        #"Human": (0.965, 0.985, 0.960),
    },
    "PP": {
        #"CLIP-ViT": (0.405, 0.13),
        #"CLIP-RN50": (0.555, 0.51, 0.295),
        #"CLIP-RN101": (0.45, 0.495, 0.23),
        #"SigLIP": (0.9, 0.005),
        #"SigLIP2": (0.555, 0.57),
        #"MetaCLIP": (0.67, 0.18),
        "MetaCLIP2": (0.725, 0.08),
        #"OpenCLIP-ViT": (0.485, 0.215),
        #"OpenCLIP-convnext": (0.515, 0.51, 0.265),
        #"EVA-CLIP": (0.545, 0.495, 0.275),
        #"Human": (0.925, 0.970, 0.915),
    },
    "Anaph": {
        #"CLIP-ViT": (0.687, 0.224),
        #"CLIP-RN50": (0.552, 0.517, 0.309),
        #"CLIP-RN101": (0.527, 0.488, 0.279),
        #"SigLIP": (0.881, 0.03),
        #"SigLIP2": (0.473, 0.672),
        #"MetaCLIP": (0.512, 0.095),
        "MetaCLIP2": (0.602, 0.065),
        #"OpenCLIP-ViT": (0.413, 0.129),
        #"OpenCLIP-convnext": (0.522, 0.508, 0.299),
        #"EVA-CLIP": (0.517, 0.508, 0.313),
        #"Human": (0.945, 0.881, 0.851),
    },
    "Ellip": {
        #"CLIP-ViT": (0.634, 0.282),
        #"CLIP-RN50": (0.525, 0.53, 0.307),
        #"CLIP-RN101": (0.525, 0.5, 0.277),
        #"SigLIP": (0.51, 0.04),
        #"SigLIP2": (0.594, 0.559),
        #"MetaCLIP": (0.663, 0.203),
        "MetaCLIP2": (0.881, 0.188),
        #"OpenCLIP-ViT": (0.792, 0.342),
        #"OpenCLIP-convnext": (0.555, 0.535, 0.337),
        #"EVA-CLIP": (0.579, 0.52, 0.297),
        #"Human": (0.896, 0.926, 0.847),
    },
    "Adj": {
        #"CLIP-ViT": (0.57, 0.16),
        #"CLIP-RN50": (0.55, 0.5, 0.295),
        #"CLIP-RN101": (0.585, 0.505, 0.295),
        #"SigLIP": (0.855, 0.025),
        #"SigLIP2": (0.645, 0.485),
        #"MetaCLIP": (0.685, 0.135),
        "MetaCLIP2": (0.785, 0.115),
        #"OpenCLIP-ViT": (0.575, 0.225),
        #"OpenCLIP-convnext": (0.615, 0.515, 0.305),
        #"EVA-CLIP": (0.585, 0.525, 0.31),
        #"Human": (0.895, 0.905, 0.855),
    },
    "Vb": {
        #"CLIP-ViT": (0.29, 0.215),
        #"CLIP-RN50": (0.555, 0.51, 0.26),
        #"CLIP-RN101": (0.58, 0.51, 0.315),
        #"SigLIP": (0.745, 0.025),
        #"SigLIP2": (0.475, 0.505),
        #"MetaCLIP": (0.55, 0.385),
        "MetaCLIP2": (0.54, 0.3),
        #"OpenCLIP-ViT": (0.36, 0.285),
        #"OpenCLIP-convnext": (0.6, 0.52, 0.31),
        #"EVA-CLIP": (0.605, 0.510, 0.365),
        #"Human": (0.945, 0.920, 0.915),
    },
    "Conj": {
        #"CLIP-ViT": (0.627, 0.173),
        #"CLIP-RN50": (0.507, 0.36, 0.197),
        #"CLIP-RN101": (0.54, 0.353, 0.187),
        #"SigLIP": (0.873, 0.033),
        #"SigLIP2": (0.2, 0.577),
        #"MetaCLIP": (0.577, 0.39),
        "MetaCLIP2": (0.803, 0.103),
        #"OpenCLIP-ViT": (0.55, 0.347),
        #"OpenCLIP-convnext": (0.557, 0.37, 0.217),
        #"EVA-CLIP": (0.54, 0.387, 0.203),
        #"Human": (0.933, 0.93, 0.913),
    },
    "All": {
        #"CLIP-ViT": (0.565, 0.192),
        #"CLIP-RN50": (0.536, 0.479, 0.272),
        #"CLIP-RN101": (0.527, 0.471, 0.256),
        #"SigLIP": (0.805, 0.03),
        #"SigLIP2": (0.478, 0.577),
        #"MetaCLIP": (0.635, 0.23),
        "MetaCLIP2": (0.734, 0.129),
        #"OpenCLIP-ViT": (0.55, 0.252),
        #"OpenCLIP-convnext": (0.558, 0.486, 0.285),
        #"EVA-CLIP": (0.551, 0.485, 0.286),
        #"Human": (0.929, 0.931, 0.896),
    },
}

# ---------------- GRID PLOT ----------------
fig, axes = plt.subplots(
    nrows=len(models),
    ncols=len(types),
    figsize=(28, 10),                      # ↓ MUCH more compact
    gridspec_kw={"height_ratios": [1]*len(models)}
)

if len(models) == 1:
    axes = np.expand_dims(axes, axis=0)


bar_width = 0.5
x = np.arange(2)   # vs Ambiguous, vs Distractor

for i, model in enumerate(models):
    for j, amb_type in enumerate(types):
        ax = axes[i, j]

        vals = np.array(results[amb_type][model]) * 100  # convert to %
        ax.bar(x, vals, color=["#830F0F", "#0B0C0B"], width=bar_width)

        # Title only on top row
        if i == 0:
            ax.set_title(amb_type, fontsize=16, fontweight="bold", pad=10)

        # Model label only on left column
        if j == 0:
            ax.set_ylabel(model, fontsize=14, fontweight="bold",
                          rotation=0, labelpad=28, va="center")

        # Proper y-limit & ticks
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis='y', labelsize=8)

        # No x-axis ticks
        ax.set_xticks([])

        # Baseline: 50%
        ax.axhline(50, color='red', linestyle='-', linewidth=1.2)

        # Remove top/right borders
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)

# Global legend
fig.legend(
    ["Threshold", "vs Ambiguous Caption", "vs Distracting Caption"],
    loc='upper center', ncol=3, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.98)
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
plt.savefig("versus_grid.png", dpi=300, bbox_inches="tight")
plt.show()
