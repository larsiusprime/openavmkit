"""
Kolbe, et al.
-----------
Implementation of the Kolbe et al. (2023) paper.

**Experimental and WIP - not yet ready for production use.**
"""

from scipy.spatial._ckdtree import cKDTree
from scipy.special import comb

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import trange

from hilbertcurve.hilbertcurve import HilbertCurve

from openavmkit.data import (
    SalesUniversePair,
    get_hydrated_sales_from_sup,
    get_sale_field,
)
from openavmkit.utilities.data import div_df_z_safe


def difference_weights(m: int) -> np.ndarray:
    """
    Return Δₘ weights that satisfy Σw = 0 and ‖w‖₂ = 1.

    Parameters
    ----------
    m : int
        Number of weights to generate.

    Returns
    -------
    np.ndarray
        Array of length `m` whose elements sum to zero and whose L2 norm is one.
    """
    w = np.array([(-1) ** s * comb(m, s) for s in range(m + 1)], dtype=float)
    return w / np.linalg.norm(w)


def hilbert_order(lat: np.ndarray, lon: np.ndarray, n_bits: int = 16) -> np.ndarray:
    """
    Return indices that sort (lat, lon) via a Hilbert curve.

    Parameters
    ----------
    lat : np.ndarray
        Array of latitude values.
    lon : np.ndarray
        Array of longitude values.
    n_bits : int, optional
        Number of bits for Hilbert curve resolution. Defaults to 16.

    Returns
    -------
    np.ndarray
        Array of indices that sort points along the Hilbert curve.
    """

    if lat.size != lon.size:
        raise ValueError("lat and lon must have the same length")

    # Scale each axis to the integer grid [0, 2ⁿ_bits‑1]
    lat_scaled = (
        (lat - lat.min()) / (lat.max() - lat.min()) * (2**n_bits - 1)
    ).astype(int)
    lon_scaled = (
        (lon - lon.min()) / (lon.max() - lon.min()) * (2**n_bits - 1)
    ).astype(int)

    coords_int = np.stack((lat_scaled, lon_scaled), axis=1).tolist()

    hc = HilbertCurve(n_bits, 2)
    dist_1d = hc.distances_from_points(coords_int, match_type=True)

    return np.argsort(dist_1d)


def adaptive_weights_smoothing(
    resid: np.ndarray,
    coords: np.ndarray,
    *,
    k_neighbors: int,
    h0: float = 500.0,
    n_iter: int = 6,
    alpha: float = 0.6,
    verbose: bool = False,
) -> np.ndarray:
    """
    Perform adaptive weights smoothing on residuals using spatial coordinates.

    This function computes smoothing weights for each observation based on its residual value
    and the distances to its k nearest neighbors. The initial bandwidth `h0` is iteratively
    updated over `n_iter` iterations according to the adaptive smoothing parameter `alpha`.

    Parameters
    ----------
    resid : np.ndarray
        Array of residual values for each observation (shape `(n,)`).
    coords : np.ndarray
        Array of spatial coordinates for each observation (shape `(n, 2)`).
    k_neighbors : int
        Number of nearest neighbors to consider when computing local weights.
    h0 : float, optional
        Initial bandwidth for smoothing (distance threshold), by default 500.0.
    n_iter : int, optional
        Number of iterations to perform for adaptive bandwidth adjustment, by default 6.
    alpha : float, optional
        Adaptation rate for updating the bandwidth at each iteration, by default 0.6.
    verbose : bool, optional
        If True, print progress information during iterations, by default False.

    Raises
    ------
    ValueError
        If `k_neighbors` is not a positive integer.

    Returns
    -------
    np.ndarray
        Array of smoothed weights (shape `(n,)`), normalized so that they sum to one.
    """

    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")

    tree = cKDTree(coords)
    dists, neigh_idx = tree.query(coords, k=k_neighbors)

    finite = resid[~np.isnan(resid)]
    sigma = np.nanmedian(np.abs(finite - np.nanmedian(finite))) / 0.6745
    sigma = max(float(sigma), 1e-8)

    a_hat = np.where(np.isnan(resid), float(np.nanmedian(finite)), resid)
    h = np.full_like(resid, h0, dtype=float)

    for _ in trange(n_iter, disable=not verbose, desc="AWS k‑NN"):
        new_vals = np.empty_like(a_hat)
        for i in range(resid.size):
            idx = neigh_idx[i]
            mask = ~np.isnan(resid[idx])
            if not mask.any():
                new_vals[i] = a_hat[i]
                continue
            d = dists[i][mask]
            r_j = resid[idx][mask]
            a_j = a_hat[idx][mask]

            K_dist = np.exp(-(d**2) / (2.0 * h[i] ** 2))
            K_sim = np.exp(-((a_hat[i] - a_j) ** 2) / (2.0 * (alpha * sigma) ** 2))
            w = K_dist * K_sim
            w_sum = w.sum()
            new_vals[i] = a_hat[i] if w_sum == 0 else np.dot(w, r_j) / w_sum

            flatness = np.mean(np.abs(a_hat[i] - a_j)) if w_sum else 0.0
            h[i] = np.clip(h[i] * (1 - 0.5 * flatness / (alpha * sigma)), h0 * 0.3, h0)
        a_hat = new_vals
    return a_hat


def kolbe_et_al_estimate(
    sup: SalesUniversePair,
    bldg_fields: list[str],
    model_group: str,
    settings: dict,
    params: dict = None,
    verbose: bool = False,
) -> tuple[sm.regression.linear_model.RegressionResults, pd.Series]:
    """
    Estimate adaptive weights smoothing (AWS) residuals using the Kolbe et al. (2023) method for a given model group.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data pair containing sales data and universe information.
    bldg_fields : list[str]
        List of building fields to include in the estimation.
    model_group : str
        The model group to filter the sales and universe data.
    settings : dict
        Settings dictionary containing configuration parameters.
    params : dict, optional
        Dictionary of parameters for the estimation, including:
    verbose : bool, optional
        If True, print progress information during iterations, by default False.

    Returns
    -------
    tuple[sm.regression.linear_model.RegressionResults, pd.Series]
        A tuple containing:
        - Regression results from the OLS model.
        - Series representing adaptive weights smoothing residuals.
    """
    if params is None:
        params = {}

    k_neighbors = params.get("k_neighbors", 60)
    diff_order = params.get("diff_order", 10)
    h0 = params.get("pilot_bandwidth", 600.0)
    n_iter = params.get("n_iter", 4)

    df_sales = get_hydrated_sales_from_sup(sup)
    df_univ = sup.universe

    # Select the model group:
    df_sales = df_sales[df_sales["model_group"].eq(model_group)].copy()
    df_univ = df_univ[df_univ["model_group"].eq(model_group)].copy()

    sale_field = get_sale_field(settings)

    # ensure we have the columns we need:
    necessary_cols = [
        sale_field,
        "latitude",
        "longitude",
        "land_area_sqft",
    ] + bldg_fields
    for field in necessary_cols:
        if field not in df_sales.columns:
            raise ValueError(f"Sales dataframe must have a '{field}' field")
        if field not in df_univ.columns:
            if field == sale_field:
                df_univ[field] = None
            else:
                raise ValueError(f"Universe dataframe must have a '{field}' field")

    # ------------------------------------------
    # 0. Construct the DataFrame
    # ------------------------------------------

    # Get only the fields we care about:
    df_univ = df_univ[["key"] + necessary_cols]

    # DF base is the sales dataframe
    df = df_sales[["key", "key_sale", "sale_date"] + necessary_cols].copy()

    # Determine which keys are not in sales but are in univ:
    df_univ_to_add = df_univ[~df_univ["key"].isin(df["key"])].copy()

    df_univ_to_add["key_sale"] = None
    df_univ_to_add["sale_date"] = None

    # Add the missing rows from df_univ_to_add to df:
    df = pd.concat([df, df_univ_to_add], ignore_index=True)
    df = df[~df["latitude"].isna() & ~df["longitude"].isna()]

    # ----------------------------------------------
    # 1. Convert to price-per-area and building-per-area
    # ----------------------------------------------

    df["p"] = div_df_z_safe(df, sale_field, "land_area_sqft")
    p_area_cols: list[str] = []
    for col in bldg_fields:
        p_area = f"{col}_per_land_sqft"
        df[p_area] = div_df_z_safe(df, col, "land_area_sqft")
        p_area_cols.append(p_area)

    # ---------------------------------------------
    # 2. Spatial ordering
    # ---------------------------------------------

    order = hilbert_order(df["latitude"].values, df["longitude"].values)
    df = df.iloc[order].reset_index(drop=True)

    # ----------------------------------------------
    # 3. Higher‑order differences
    # ----------------------------------------------

    d = difference_weights(diff_order)

    def diff_series(s: pd.Series) -> pd.Series:
        X = np.column_stack([s.shift(k) for k in range(diff_order + 1)])
        return pd.Series(X[diff_order:] @ d, index=s.index[diff_order:])

    y_d = diff_series(df["p"])
    X_d = pd.DataFrame({c: diff_series(df[c]) for c in p_area_cols})
    X_d = sm.add_constant(X_d, has_constant='add')

    # drop rows with NaNs (unsold rows propagate NaN through differences)
    valid = y_d.notna() & X_d.notna().all(axis=1)
    y_d = y_d[valid]
    X_d = X_d.loc[valid]

    ols_res = sm.OLS(y_d, X_d, hasconst=True).fit(cov_type="HC1")

    # ----------------------------------------------
    # 4. AWS residual smoothing
    # ----------------------------------------------

    resid = df["p"].iloc[diff_order:] - (
        df.loc[diff_order:, p_area_cols] @ ols_res.params[p_area_cols]
        + ols_res.params["const"]
    )
    resid_np = resid.to_numpy(dtype=float)  # ensure np.nan not pd.NA

    # convert lat/lon to planar metres (equirectangular)
    R = 6_371_000.0
    lat_rad = np.radians(df["latitude"].iloc[diff_order:].values)
    lon_rad = np.radians(df["longitude"].iloc[diff_order:].values)
    x = R * lon_rad * np.cos(lat_rad.mean())
    y = R * lat_rad
    coords_m = np.column_stack([x, y])

    a_hat = adaptive_weights_smoothing(
        resid_np,
        coords_m,
        k_neighbors=k_neighbors,
        h0=h0,
        n_iter=n_iter,
        verbose=verbose,
    )

    return ols_res, pd.Series(a_hat, index=df.index[diff_order:], name="a_hat")
