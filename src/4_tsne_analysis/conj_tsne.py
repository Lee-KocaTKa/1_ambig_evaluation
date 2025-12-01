#!/usr/bin/env python
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")  # for cluster / non-GUI environments
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ----------------------------
# 1. Feature extraction helpers
# ----------------------------
def normalize_vec(vec, expected_dim=768):
    """
    Convert input vec into a flat 1D numpy array of length expected_dim.
    Returns None if shape is invalid.
    Handles:
      - list of floats
      - [ [floats...] ]
      - numpy arrays with extra dims
    """
    if vec is None:
        print("None")
        return None

    # If nested like [[...]]
    if isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
        print("Nested")
        vec = vec[0]

    # Convert to numpy
    vec = np.array(vec, dtype=float).reshape(-1)



    if vec.shape[0] == expected_dim:
        return vec
    else:
        print(vec.shape[0])
        return None


def extract_features(data, expected_dim=768):
    """
    Extract ambiguous and variant features from the loaded pickle list.
    Returns:
      - vectors: numpy array (N, expected_dim)
      - labels: list of str labels for each vector
      - ids:    list of sample indices (linking back to original data)
    """

    all_vectors = []
    all_labels = []
    all_ids = []

    # keys to extract from each sample
    variant_keys = [
        "Ambiguous_Text_Feature",
        "Variant_0_Text_Feature",
        "Variant_1_Text_Feature",
        "Variant_2_Text_Feature",  # enable if present
        "Variant_0_Image_Feature",
        "Variant_1_Image_Feature",
        "Variant_2_Image_Feature",  # enable if present
    ]

    for idx, sample in enumerate(data):
        for key in variant_keys:
            if key not in sample:
                continue
            
            
            
            vec = normalize_vec(sample[key], expected_dim=expected_dim)
            if vec is None:
                print(f"[WARN] Invalid vector for {key} in sample {idx}")
                continue

            all_vectors.append(vec)
            all_labels.append(key)
            all_ids.append(idx)

    if len(all_vectors) == 0:
        raise ValueError("[ERROR] No valid vectors extracted. Check keys and expected_dim.")

    all_vectors = np.stack(all_vectors, axis=0)
    print(f"[INFO] Extracted {all_vectors.shape[0]} vectors of dimension {all_vectors.shape[1]}")
    return all_vectors, all_labels, all_ids


# ----------------------------
# 2. t-SNE (2D)
# ----------------------------
def run_tsne_2d(vectors, seed=42, perplexity=30.0):
    """
    Run 2D t-SNE on the given vectors.
    """
    print("[INFO] Running t-SNE (2D)...")
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
    )
    embedding = tsne.fit_transform(vectors)
    print("[INFO] t-SNE done.")
    return embedding


# ----------------------------
# 3. Plotting (2D)
# ----------------------------
def plot_tsne_2d(embedding, labels, save_path="tsne_2d.png", title="t-SNE (2D)"):
    """
    Plot a 2D t-SNE embedding with points colored by label.
    """
    print(f"[INFO] Plotting t-SNE to {save_path} ...")
    plt.figure(figsize=(10, 8))

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for lab, col in zip(unique_labels, colors):
        idxs = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(
            embedding[idxs, 0],
            embedding[idxs, 1],
            s=18,
            color=col,
            alpha=0.75,
            label=lab,
        )

    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved 2D t-SNE to {save_path}")


# ----------------------------
# 4. Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-pkl",
        type=str,
        default="../../../data/ViLStrUB/classification_results/results_vit-openclip_conj.pkl",
        help="Path to the pickle file (list of dicts with features).",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default="tsne_openclip_vp_2d.png",
        help="Path to output PNG file.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=768,
        help="Expected dimensionality of feature vectors.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity parameter for t-SNE.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Optional: randomly subsample to at most this many vectors for speed (0 = no subsampling).",
    )
    args = parser.parse_args()

    # 1. Load data
    print(f"[INFO] Loading data from {args.input_pkl}")
    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)

    # 2. Extract features
    vectors, labels, ids = extract_features(data, expected_dim=args.expected_dim)

    # Optional subsampling if too many points
    if args.max_samples > 0 and len(vectors) > args.max_samples:
        print(f"[INFO] Subsampling from {len(vectors)} to {args.max_samples} for t-SNE...")
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(vectors), size=args.max_samples, replace=False)
        vectors = vectors[idxs]
        labels = [labels[i] for i in idxs]
        ids = [ids[i] for i in idxs]

    # 3. Run t-SNE
    embedding = run_tsne_2d(vectors, perplexity=args.perplexity)

    # 4. Plot
    title = os.path.splitext(os.path.basename(args.output_png))[0]
    plot_tsne_2d(embedding, labels, save_path=args.output_png, title=title)


if __name__ == "__main__":
    main()
