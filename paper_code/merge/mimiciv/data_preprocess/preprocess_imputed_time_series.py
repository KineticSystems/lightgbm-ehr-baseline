from tqdm import tqdm


def compute_global_means(irg_labs_vitals_df, stay_ids, feature_cols):
    """Compute train-only fallback means for labs/vitals features."""
    subset = irg_labs_vitals_df[irg_labs_vitals_df['stay_id'].isin(stay_ids)]
    means = subset[feature_cols].mean()
    nan_features = means[means.isna()].index.tolist()
    if nan_features:
        raise ValueError(
            f"Train split has no observed values for features: {nan_features}. "
            "Cannot compute fallback means."
        )
    return means


def impute_stays(irg_labs_vitals_df, stay_ids, global_means, feature_cols, seq_len, interval_length=1):
    """Impute labs/vitals for stays using train-derived fallback means.

    Returns a dict mapping stay_id to a regular feature-only array of shape
    (seq_len, len(feature_cols)). Stays with no labs/vitals rows are omitted.
    """
    if seq_len is None or seq_len <= 0:
        raise ValueError(f"seq_len must be a positive integer, got {seq_len}")

    missing_mean_cols = [col for col in feature_cols if col not in global_means.index]
    if missing_mean_cols:
        raise ValueError(f"Global means missing feature columns: {missing_mean_cols}")

    global_means = global_means.reindex(feature_cols)
    stays_df = irg_labs_vitals_df[irg_labs_vitals_df['stay_id'].isin(stay_ids)]
    imputed = {}

    for stay_id, group in tqdm(stays_df.groupby('stay_id', sort=False), desc="Imputing stays"):
        if len(group) == 0:
            continue

        group = group.copy()
        group['time_interval'] = (group['icu_time_delta'] / interval_length).astype(int)

        curr_stay_imputed = group.groupby('time_interval')[feature_cols].mean()
        curr_stay_imputed = curr_stay_imputed.reindex(range(seq_len))
        curr_stay_imputed = curr_stay_imputed.interpolate(
            method='linear',
            limit_direction='both',
        )
        curr_stay_imputed = curr_stay_imputed.fillna(global_means)

        if curr_stay_imputed[feature_cols].isna().any().any():
            nan_features = curr_stay_imputed.columns[curr_stay_imputed.isna().any()].tolist()
            raise ValueError(f"Imputation left NaNs for stay {stay_id}, features: {nan_features}")

        imputed[stay_id] = curr_stay_imputed[feature_cols].to_numpy()

    return imputed
