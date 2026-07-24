import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import os
from TF_attributes_fs import compute_attributes
import pickle

ATTR_DICT = compute_attributes()


def extract_features(signal):
    """Extract all attribute features from a vibration signal."""
    features = {}
    for name, func in ATTR_DICT.items():
        try:
            features[name] = float(func(signal))
        except Exception:
            features[name] = 0.0
    return features


def mi_knn_mixed(
        X, y,
        alpha: float = 5.0,
        eps: float = 1e-12
):
    """
    Estimate Mutual Information I(X; y) using adaptive radius kNN.

    The discriminative term I(F):
    I(F) = H(y) - H(y|X), where H(y|X) is estimated via local conditional
    entropy using adaptive neighborhoods based on 1st nearest neighbor distance.

    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    N = X.shape[0]
    C = int(y.max() + 1)

    #  Entropy of fault classes H(y)
    class_counts = np.bincount(y, minlength=C)
    p_y = class_counts[class_counts > 0] / N
    H_y = -np.sum(p_y * np.log2(p_y + eps))

    # Build KDTree and compute 1st NN distances
    tree = KDTree(X)
    dists, _ = tree.query(X, k=2)  # dists[:,0] is self, [:,1] is r_{1,n}
    r1 = dists[:, 1] + eps
    r1_mean = r1.mean()

    #  Adaptive radius β_n
    beta_n = alpha * (r1_mean / r1)
    beta_n = np.clip(beta_n, 1.0, 30.0)  # reasonable bounds for stability

    # Local conditional entropy H(y|X)
    H_cond_local = np.zeros(N)

    for i in range(N):
        radius = r1[i] * beta_n[i]

        # Query all points within adaptive radius
        idxs = tree.query_ball_point(X[i], r=radius)
        if len(idxs) < 2:
            # fallback for edge cases
            idxs = tree.query_ball_point(X[i], r=r1[i] * 2)

        cnts = np.bincount(y[idxs], minlength=C)
        total = len(idxs)

        if total <= 1:
            H_cond_local[i] = 0.0
        else:
            p_local = cnts / total
            p_local = p_local[p_local > 0]
            H_cond_local[i] = -np.sum(p_local * np.log2(p_local + eps))

    H_y_given_X = H_cond_local.mean()

    #  Mutual Information
    I = max(0.0, H_y - H_y_given_X)

    return float(I), float(H_y), float(H_y_given_X)


def calculate_effective_space(features_df, fault_labels, selected_features):
    """
    Compute Effective Discriminative Information (EDI) score.

    EDI(F) = I(F) × Q̄_eff(F)
    - I(F): discriminative power (mutual information)
    - Q̄_eff(F): average effective coverage across features

    """
    if len(selected_features) == 0:
        return 0.0, 0.0, 0.0, {}

    features_df = pd.DataFrame(features_df)
    y = np.asarray(fault_labels).astype(int)
    N = len(y)

    valid_features = [f for f in selected_features if f in features_df.columns]
    if len(valid_features) == 0:
        return 0.0, 0.0, 0.0, {f: 0.0 for f in selected_features}

    X_raw = features_df[valid_features].values.astype(float)

    #  Normalize each feature to [0,1]
    X_norm = np.zeros_like(X_raw)
    for j in range(X_raw.shape[1]):
        col = X_raw[:, j]
        mn, mx = col.min(), col.max()
        if mx > mn:
            X_norm[:, j] = (col - mn) / (mx - mn)
        else:
            X_norm[:, j] = 0.0
    X_norm = np.clip(X_norm, 0.0, 1.0)

    #  Discriminative Term I(F)
    if N < 5 or X_norm.shape[1] == 0:
        I_joint = 0.0
    else:
        I_joint, _, _ = mi_knn_mixed(X_norm, y, alpha=5.0)

    # Effective Coverage Term Q_eff
    B = max(10, int(np.ceil(np.log2(N) + 2)))  # Sturges' rule
    beta = (N - 1.0) / N if N > 1 else 0.0

    Q_values = {}
    Q_sum = 0.0

    for j, fname in enumerate(valid_features):
        col = X_norm[:, j]
        hist, _ = np.histogram(col, bins=B, range=(0.0, 1.0))
        n_b = hist.astype(float)

        # Effective activated samples per bin
        activated_bins = np.sum(n_b > 0)
        Q_eff = activated_bins / B if B > 0 else 0.0

        # Q_eff(F_m) - average over bins
        Q_values[fname] = Q_eff
        Q_sum += Q_eff

    avg_Q = Q_sum / len(valid_features)

    #  Final EDI Score
    edi_score = float(I_joint * avg_Q)

    # Fill missing features with 0
    for f in selected_features:
        if f not in Q_values:
            Q_values[f] = 0.0

    return edi_score, I_joint, avg_Q, Q_values


#  Data Loading Functions

def get_cache_paths(folder_path):
    """Return cache file paths for features and labels."""
    cache_df = os.path.join(folder_path, "feature_cache.parquet")
    cache_label = os.path.join(folder_path, "label_cache.npy")
    cache_class = os.path.join(folder_path, "classname_cache.npy")
    return cache_df, cache_label, cache_class


def build_feature_cache(
    folder_path,
    sample_length=1024,
    step=512,
    max_samples_per_csv=100,
    cache_path="feature_cache.pkl"
):
    """
    Scan all CSV files in the folder, extract features and save to cache
    """
    features_list = []
    labels = []
    csv_names = []

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    csv_files.sort()

    csv_to_label = {name: idx for idx, name in enumerate(csv_files)}

    for csv_name in csv_files:
        file_path = os.path.join(folder_path, csv_name)
        df = pd.read_csv(file_path)
        signal = df.iloc[:, 0].values

        start = 0
        cnt = 0
        while start + sample_length <= len(signal) and cnt < max_samples_per_csv:
            segment = signal[start:start + sample_length]
            start += step

            feats = extract_features(segment)
            feats["csv_name"] = csv_name
            feats["label"] = csv_to_label[csv_name]

            features_list.append(feats)
            cnt += 1

        print(f"Cached {csv_name}: {cnt} samples")

    cache_df = pd.DataFrame(features_list)

    with open(cache_path, "wb") as f:
        pickle.dump({
            "data": cache_df,
            "csv_to_label": csv_to_label
        }, f)

    print(f"\nCache completed：{cache_df.shape[0]} samples")


def load_from_cache(cache_path, selected_csvs):

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    df = cache["data"]

    df_sel = df[df["csv_name"].isin(selected_csvs)].copy()

    feature_cols = [c for c in df_sel.columns if c not in ("csv_name", "label")]

    features_df = df_sel[feature_cols]
    fault_labels = df_sel["label"].values

    class_names = selected_csvs

    print(f"Loaded from cache: {len(features_df)} samples, {len(class_names)} classes")

    return features_df, fault_labels, class_names


def get_feature_data_with_cache(
    folder_path,
    selected_csvs,
    cache_path="feature_cache.pkl",
    sample_length=512,
    step=64,
    max_samples_per_csv=300
):

    if not os.path.exists(cache_path):
        print("Feature cache not found, building feature_cache.pkl ...")
        build_feature_cache(
            folder_path=folder_path,
            sample_length=sample_length,
            step=step,
            max_samples_per_csv=max_samples_per_csv,
            cache_path=cache_path
        )
    else:
        print("Feature cache detected, loading directly")

    # Load required CSV data uniformly from cache
    features_df, fault_labels, class_names = load_from_cache(
        cache_path=cache_path,
        selected_csvs=selected_csvs
    )

    return features_df, fault_labels, class_names
