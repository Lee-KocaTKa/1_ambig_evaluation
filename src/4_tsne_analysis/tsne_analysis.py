import pickle
import numpy as np
import os 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


with open("../../../data/ViLStrUB/classification_results/results_metaclip2_coordinate.pkl", "rb") as f:
    data = pickle.load(f)   # data is a list of dicts


def extract_features(data, expected_dim=768):
    all_vectors = []
    all_labels = []
    all_ids = []

    variant_keys = [
        "Variant_0_Text_Feature",
        "Variant_1_Text_Feature",
        "Variant_2_Text_Feature",
        "Variant_0_Image_Feature",
        "Variant_1_Image_Feature",
        "Variant_2_Image_Feature",
    ]

    for idx, sample in enumerate(data):

        # ---- Handle ambiguous sentence ----
        if "Ambiguous_Text_Feature" in sample:
            vec = sample["Ambiguous_Text_Feature"]

            # Ambiguous_Sentence is stored as: [ [768 floats] ]
            if isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
                vec = vec[0]

            # sanity check
            if isinstance(vec, list) and len(vec) == expected_dim:
                all_vectors.append(vec)
                all_labels.append("Ambiguous")
                all_ids.append(idx)
            else:
                print(f"[WARN] Ambiguous vector wrong shape at sample {idx}: {len(vec) if isinstance(vec,list) else type(vec)}")

        # ---- Handle variants ----
        for key in variant_keys:
            if key in sample:
                vec = sample[key]

                # ensure ambiguous nested lists
                if isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
                    vec = vec[0]

                # ensure correct dim
                if isinstance(vec, list) and len(vec) == expected_dim:
                    all_vectors.append(vec)
                    all_labels.append(key)
                    all_ids.append(idx)
                else:
                    print(f"[WARN] Bad vector for {key} in sample {idx}: length={len(vec) if isinstance(vec,list) else type(vec)}")

    # convert to clean NumPy array
    all_vectors = np.array(all_vectors, dtype=float)
    print(f"[INFO] Extracted {len(all_vectors)} vectors of dimension {all_vectors.shape[1]}")
    return all_vectors, all_labels, all_ids


def run_tsne(vectors, n_components=3, seed=42):
    tsne = TSNE(
        n_components=n_components,
        random_state=seed,
        perplexity=30,
        learning_rate="auto",
        init="pca"
    )
    embedding = tsne.fit_transform(vectors)
    return embedding




def plot_tsne(embedding, labels, save_path=".", title="metaclip2-conjunction"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # consistent label-color mapping
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for lab, col in zip(unique_labels, colors):
        idxs = [i for i, l in enumerate(labels) if l == lab]
        ax.scatter(
            embedding[idxs, 0],
            embedding[idxs, 1],
            embedding[idxs, 2],
            s=18,
            color=col,
            label=lab,
            depthshade=True
        )

    # ===== Shrink axes to remove empty space =====
    x_min, x_max = embedding[:,0].min(), embedding[:,0].max()
    y_min, y_max = embedding[:,1].min(), embedding[:,1].max()
    z_min, z_max = embedding[:,2].min(), embedding[:,2].max()
    pad = 0.05

    ax.set_xlim([x_min - (x_max-x_min)*pad, x_max + (x_max-x_min)*pad])
    ax.set_ylim([y_min - (y_max-y_min)*pad, y_max + (y_max-y_min)*pad])
    ax.set_zlim([z_min - (z_max-z_min)*pad, z_max + (z_max-z_min)*pad])

    # Title & labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.tight_layout()

    if save_path:
        #os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved 3D t-SNE to {save_path}")
    else:
        plt.show()




# 2. extract features
vectors, labels, ids = extract_features(data)

# 3. run tsne
embedding = run_tsne(vectors)

# 4. plot
plot_tsne(embedding, labels)
