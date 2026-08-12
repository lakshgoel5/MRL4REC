"""
Prepare feature-based initialization embeddings for MRL4REC.

Loads original rich features from tripartite_aug_ttv/, builds global
user and item feature matrices aligned with MRL4REC's index space,
applies PCA to project to the target embedding dimension, and saves
user_init.npy and item_init.npy.

Usage:
    python prepare_feature_init.py \
        --src_dir /path/to/tripartite_aug_ttv/ \
        --out_dir /path/to/output/ \
        --emb_size 768

The emb_size must match --dim used when training MRL4REC.

User ID ordering assumed (must match how the MRL4REC dataset was built):
    train users: global IDs 0    – 6999  (train/ue.npy)
    test  users: global IDs 7000 – 7999  (test/ue.npy)
    val   users: global IDs 8000 – 9999  (val/ue.npy)

Item IDs are mapped via product_mapping_10k.csv, exactly as in
extract_model_emb.py, so the output aligns with MRL4REC's item vocab.
"""

import argparse
import os
import numpy as np
import pandas as pd


# Global user ID boundaries per split — must match extract_model_emb.py
SPLITS = [
    ("train", 0,    7000),
    ("test",  7000, 8000),
    ("val",   8000, 10000),
]


def build_init_embeddings(src_data_dir: str, out_dir: str, emb_size: int):
    # ------------------------------------------------------------------ #
    # 1.  User features: stack splits in global-ID order                 #
    # ------------------------------------------------------------------ #
    user_parts = []
    for split_name, start, end in SPLITS:
        path = os.path.join(src_data_dir, split_name, "ue.npy")
        ue = np.load(path).astype(np.float32)
        if ue.shape[0] != end - start:
            raise ValueError(
                f"{split_name}/ue.npy: expected {end - start} rows, got {ue.shape[0]}"
            )
        user_parts.append(ue)

    user_features = np.vstack(user_parts)          # [n_users, feat_dim]
    print(f"User features stacked: {user_features.shape}")

    # ------------------------------------------------------------------ #
    # 2.  Item features: build global vocabulary (mirrors extract_model_emb.py) #
    # ------------------------------------------------------------------ #
    all_canonical_products = set()
    split_dfs = {}
    for split_name, _, _ in SPLITS:
        csv_path = os.path.join(src_data_dir, split_name, "product_mapping_10k.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing: {csv_path}")
        df = pd.read_csv(csv_path)
        split_dfs[split_name] = df
        all_canonical_products.update(df["product_id"].unique())

    sorted_products = sorted(all_canonical_products)
    n_items = len(sorted_products)
    str2mrl_id = {p: i for i, p in enumerate(sorted_products)}
    print(f"Global item vocabulary: {n_items} items")

    feat_dim = user_features.shape[1]
    item_features = np.zeros((n_items, feat_dim), dtype=np.float32)
    item_filled = np.zeros(n_items, dtype=bool)

    for split_name, _, _ in SPLITS:
        pe = np.load(os.path.join(src_data_dir, split_name, "pe.npy")).astype(np.float32)
        local_to_str = split_dfs[split_name].set_index("index")["product_id"].to_dict()
        for local_idx, prod_str in local_to_str.items():
            global_id = str2mrl_id[prod_str]
            if not item_filled[global_id]:
                item_features[global_id] = pe[local_idx]
                item_filled[global_id] = True

    missing = int((~item_filled).sum())
    if missing > 0:
        print(f"Warning: {missing}/{n_items} items have no features — left as zeros")
    print(f"Item features built:   {item_features.shape}")

    # ------------------------------------------------------------------ #
    # 3.  Project to emb_size via PCA (fit jointly so U/I share subspace) #
    # ------------------------------------------------------------------ #
    if emb_size >= feat_dim:
        print(f"emb_size ({emb_size}) >= feat_dim ({feat_dim}): padding with zeros")
        user_init = np.zeros((user_features.shape[0], emb_size), dtype=np.float32)
        user_init[:, :feat_dim] = user_features
        item_init = np.zeros((item_features.shape[0], emb_size), dtype=np.float32)
        item_init[:, :feat_dim] = item_features
    else:
        print(f"Fitting SVD-PCA {feat_dim} → {emb_size} on combined user+item features …")
        all_features = np.vstack([user_features, item_features])  # [(n_users+n_items), feat_dim]

        # Centre
        mean = all_features.mean(axis=0, keepdims=True)
        all_centered = all_features - mean

        # Truncated SVD via numpy (avoids sklearn dependency)
        # We only need the top-emb_size right singular vectors
        # Use randomised SVD for speed on large matrices
        try:
            from sklearn.utils.extmath import randomized_svd
            U, S, Vt = randomized_svd(all_centered, n_components=emb_size, random_state=0)
            print("Using randomised SVD (sklearn)")
        except ImportError:
            # Fall back to full SVD — slower but correct
            print("sklearn not found; using numpy full SVD (may be slow) …")
            _, S_full, Vt_full = np.linalg.svd(all_centered, full_matrices=False)
            S  = S_full[:emb_size]
            Vt = Vt_full[:emb_size]

        components = Vt  # [emb_size, feat_dim]

        n_total = all_centered.shape[0]
        total_var = (all_centered ** 2).sum() / (n_total - 1)
        explained_var = (S ** 2 / (n_total - 1)).sum()
        print(f"Explained variance: {explained_var / total_var:.3f}")

        user_init = ((user_features - mean) @ components.T).astype(np.float32)
        item_init = ((item_features - mean) @ components.T).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 4.  Rescale to match Xavier uniform variance so gate/GCN layers    #
    #     see embeddings at the right magnitude on step 0.               #
    # ------------------------------------------------------------------ #
    xavier_std = np.sqrt(2.0 / (emb_size + emb_size))
    for name, arr in [("user", user_init), ("item", item_init)]:
        std = arr.std()
        if std > 1e-8:
            arr *= xavier_std / std
            print(f"  {name}: rescaled std {std:.4f} → {arr.std():.4f}")

    # ------------------------------------------------------------------ #
    # 5.  Save                                                            #
    # ------------------------------------------------------------------ #
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "user_init.npy"), user_init)
    np.save(os.path.join(out_dir, "item_init.npy"), item_init)
    print(f"\nSaved: {out_dir}/user_init.npy  {user_init.shape}")
    print(f"Saved: {out_dir}/item_init.npy  {item_init.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir",  required=True,
                        help="Path to tripartite_aug_ttv/ (with train/, test/, val/ subdirs)")
    parser.add_argument("--out_dir",  required=True,
                        help="Directory to write user_init.npy and item_init.npy")
    parser.add_argument("--emb_size", type=int, default=768,
                        help="Target embedding size (must match --dim in MRL4REC training)")
    args = parser.parse_args()

    build_init_embeddings(args.src_dir, args.out_dir, args.emb_size)
