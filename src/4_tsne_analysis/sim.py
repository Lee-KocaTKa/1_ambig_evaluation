#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter


# =====================================
# 1) Utility functions
# =====================================

def normalize_vec(vec):
    """Flatten ANY nested vector to 1D numpy array and L2-normalize."""
    arr = np.array(vec, dtype=float).reshape(-1)
    return arr / (np.linalg.norm(arr) + 1e-8)

def cos(a, b):
    return float(np.dot(a, b))


# =====================================
# 2) Triplet analysis
# =====================================

def analyze_triplet(v0, v1, v2):
    v0 = normalize_vec(v0)
    v1 = normalize_vec(v1)
    v2 = normalize_vec(v2)

    sims = [
        cos(v0, v1),
        cos(v0, v2),
        cos(v1, v2)
    ]
    dists = [1 - s for s in sims]

    min_d = min(dists)
    max_d = max(dists)
    ratio = min_d / (max_d + 1e-8)

    return {
        "d01": dists[0],
        "d02": dists[1],
        "d12": dists[2],
        "min_d": min_d,
        "max_d": max_d,
        "ratio": ratio
    }


# =====================================
# 3) Main logic: triplet distances + T2I/I2T rankings
# =====================================

def main():
    INPUT = "../../../data/ViLStrUB/classification_results/results_vit-openclip_conj.pkl"  
    print("[INFO] Loading data:", INPUT)

    with open(INPUT, "rb") as f:
        data = pickle.load(f)

    text_ratios = []
    img_ratios = []
    text_min = []
    text_max = []
    img_min = []
    img_max = []

    T2I_ranks, I2T_ranks = [], []
    T2I_margins, I2T_margins = [], []

    valid = 0

    for idx, sample in enumerate(data):
        try:
            t0 = sample["Variant_0_Text_Feature"]
            t1 = sample["Variant_1_Text_Feature"]
            t2 = sample["Variant_2_Text_Feature"]

            i0 = sample["Variant_0_Image_Feature"]
            i1 = sample["Variant_1_Image_Feature"]
            i2 = sample["Variant_2_Image_Feature"]
        except KeyError:
            continue

        valid += 1

        # ----- TEXT triplet -----
        tres = analyze_triplet(t0, t1, t2)
        text_ratios.append(tres["ratio"])
        text_min.append(tres["min_d"])
        text_max.append(tres["max_d"])

        # ----- IMAGE triplet -----
        ires = analyze_triplet(i0, i1, i2)
        img_ratios.append(ires["ratio"])
        img_min.append(ires["min_d"])
        img_max.append(ires["max_d"])

        # ----- T2I ranking -----
        text_vecs = [normalize_vec(t0), normalize_vec(t1), normalize_vec(t2)]
        img_vecs  = [normalize_vec(i0), normalize_vec(i1), normalize_vec(i2)]

        # For each text variant C_k, choose correct image among 3
        for k in range(3):
            sims = [cos(text_vecs[k], img_vecs[j]) for j in range(3)]
            correct_sim = sims[k]
            sorted_sims = sorted(sims, reverse=True)

            rank = sorted_sims.index(correct_sim)
            T2I_ranks.append(rank)

            print(f"T2I {idx}-sample {k}pair : {rank}")

            margin = sorted_sims[0] - sorted_sims[1]
            T2I_margins.append(margin)

        # ----- I2T ranking -----
        for k in range(3):
            sims = [cos(img_vecs[k], text_vecs[j]) for j in range(3)]
            correct_sim = sims[k]                    # <-- 정답 similarity
            sorted_sims = sorted(sims, reverse=True)

            # 몇 등인지 찾기 (0등/1등/2등)
            rank = sorted_sims.index(correct_sim)   
            I2T_ranks.append(rank)

            print(f"I2T {idx}-sample {k}pair : {rank}")

            # margin (1등 - 2등)
            margin = sorted_sims[0] - sorted_sims[1]
            I2T_margins.append(margin)

    # =====================================
    # 4) Print statistics
    # =====================================
    print("========================================")
    print("[INFO] Valid samples:", valid)

    print("\n[TEXT triplet ratio] min/max ratio")
    print(" Mean :", np.mean(text_ratios))
    print(" Median:", np.median(text_ratios))

    print("\n[IMAGE triplet ratio] min/max ratio")
    print(" Mean :", np.mean(img_ratios))
    print(" Median:", np.median(img_ratios))

    print("\n[T2I rank distribution] (0=correct, 1=2nd, 2=worst)")
    print(Counter(T2I_ranks))

    print("\n[I2T rank distribution]")
    print(Counter(I2T_ranks))

    print("\nMean T2I margin:", np.mean(T2I_margins))
    print("Mean I2T margin:", np.mean(I2T_margins))
    print("========================================")


    # =====================================
    # 5) Histograms
    # =====================================
    plt.hist(text_ratios, bins=30)
    plt.title("Text Triplet Ratio (min_d / max_d)")
    plt.xlabel("ratio")
    plt.ylabel("count")
    plt.savefig("hist_text_triplet_ratio.png")
    plt.close()

    plt.hist(img_ratios, bins=30)
    plt.title("Image Triplet Ratio (min_d / max_d)")
    plt.xlabel("ratio")
    plt.ylabel("count")
    plt.savefig("hist_image_triplet_ratio.png")
    plt.close()

    plt.hist(T2I_margins, bins=30)
    plt.title("T2I similarity margins (1st - 2nd)")
    plt.xlabel("margin")
    plt.ylabel("count")
    plt.savefig("hist_T2I_margin.png")
    plt.close()

    plt.hist(I2T_margins, bins=30)
    plt.title("I2T similarity margins (1st - 2nd)")
    plt.xlabel("margin")
    plt.ylabel("count")
    plt.savefig("hist_I2T_margin.png")
    plt.close()

    print("[INFO] Histograms saved.")


if __name__ == "__main__":
    main()
