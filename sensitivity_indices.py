"""Sensitivity analyses I: alternative frailty indices and cohort restrictions.

Usage: python sensitivity_indices.py <dataset.xlsx> <out.xlsx> [n_boot]

Analyses of the perioperative composite (RD = OSR minus EVAR per stratum):
  - primary stratification by mFI-9 (0-1 / 2 / >=3)
  - mFI-5a stratification (0 / 1 / >=2)
  - extended index: mFI-9 count + body-mass index < 22 kg/m2 (0-1 / 2 / >=3)
  - restrictions: excluding emergency admissions, aneurysm diameter >= 50 mm, and
    excluding index repairs in 2026
  - E-value for the frail-stratum risk ratio
  - continuous mFI-9: change in RD per one-point increase (weighted linear probability
    model osr x count)

Every analysis repeats the primary method: stratum-specific propensity score,
overlap weights, percentile bootstrap (resampling within treatment x stratum,
propensity score re-estimated in each replicate). The P value for the
frail-versus-robust difference in RD uses the bootstrap standard error.
Random seed 20261012 (one generator shared sequentially by all analyses).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ow_core import COVARIATES, load, ow_weights, weighted_rd

warnings.filterwarnings("ignore")
Y = "periop_composite"
LABELS = ["Robust", "Pre-frail", "Frail"]


def strata_rd(df, strat):
    """RD (OSR - EVAR) and weighted risks per stratum; strat is a 0/1/2 Series."""
    out = []
    for k in range(3):
        d = df[strat == k]
        w = ow_weights(d)
        out.append(weighted_rd(d, Y, w))
    return out


def resample(df, strat, rng):
    idx = []
    for k in range(3):
        for t in (0, 1):
            ii = np.flatnonzero(((strat == k) & (df["osr"] == t)).to_numpy())
            idx.append(rng.choice(ii, size=len(ii), replace=True))
    idx = np.concatenate(idx)
    return df.iloc[idx].reset_index(drop=True), strat.iloc[idx].reset_index(drop=True)


def slope(df):
    """Change in RD per one-point increase in mFI-9 count: weighted linear
    probability model y ~ osr * count using the stratum-specific overlap weights."""
    w = np.empty(len(df))
    for k in range(3):
        m = (df["mFI9_cat3"] == k).to_numpy()
        w[m] = ow_weights(df[m])
    x = df["mFI9_count"].to_numpy(float)
    t = df["osr"].to_numpy(float)
    X = np.column_stack([np.ones_like(x), t, x, t * x])
    W = np.sqrt(w)
    beta = np.linalg.lstsq(X * W[:, None], df[Y].to_numpy(float) * W, rcond=None)[0]
    return beta[3]


def evalue(rr):
    rr = 1 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1)) if rr > 1 else 1.0


def run(name, df, strat, n_boot, rng):
    est = strata_rd(df, strat)
    boots = []
    for _ in range(n_boot):
        try:
            b, s = resample(df, strat, rng)
            r = strata_rd(b, s)
            boots.append([x[2] for x in r] + [r[2][1] / r[2][0] if r[2][0] > 0 else np.nan])
        except Exception:
            continue
    B = np.array(boots, dtype=float)
    rows = []
    for k in range(3):
        r0, r1, rd = est[k]
        lo, hi = np.nanpercentile(B[:, k], [2.5, 97.5])
        n = int((strat == k).sum())
        ev = (f"{int(((strat == k) & (df.osr == 0) & (df[Y] == 1)).sum())}/"
              f"{int(((strat == k) & (df.osr == 0)).sum())}",
              f"{int(((strat == k) & (df.osr == 1) & (df[Y] == 1)).sum())}/"
              f"{int(((strat == k) & (df.osr == 1)).sum())}")
        rows.append(dict(analysis=name, stratum=LABELS[k], n=n, evar_events=ev[0], osr_events=ev[1],
                         risk_evar=100 * r0, risk_osr=100 * r1,
                         rd=100 * rd, rd_lo=100 * lo, rd_hi=100 * hi))
    d = B[:, 2] - B[:, 0]
    delta = est[2][2] - est[0][2]
    se = np.nanstd(d, ddof=1)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    rows.append(dict(analysis=name, stratum="Frail - robust", rd=100 * delta, rd_lo=100 * lo,
                     rd_hi=100 * hi, p=2 * norm.sf(abs(delta) / se), n_boot=len(B)))
    # E-value for the frail stratum, risk-ratio scale (weighted risks)
    rr = est[2][1] / est[2][0] if est[2][0] > 0 else np.nan
    rr_lo = np.nanpercentile(B[:, 3], 2.5)
    return rows, dict(analysis=name, rr_frail=rr, rr_lo=rr_lo,
                      evalue_point=evalue(rr) if np.isfinite(rr) else np.nan,
                      evalue_ci=evalue(rr_lo) if rr_lo > 1 else 1.0)


def main():
    dataset, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    rng = np.random.default_rng(20261012)
    df = load(dataset).reset_index(drop=True)
    cat = lambda c, cuts: pd.Series(np.digitize(c, cuts), index=c.index)

    # alternative pre-operative indices
    df["mfi5a_cat"] = cat(df["mFI5a_count"], [1, 2])                # 0 / 1 / >=2
    # emergency admission is not part of the extended index: it marks symptomatic/urgent
    # repair and is kept as a propensity-score covariate instead
    df["ext_count"] = (df["mFI9_count"] + (df["BMI"] < 22)).astype(int)  # mFI-9 + BMI<22
    df["ext_cat"] = cat(df["ext_count"], [2, 3])                     # 0-1 / 2 / >=3

    analyses = [
        ("Primary (mFI-9)", df, df["mFI9_cat3"]),
        ("mFI-5a (0 / 1 / >=2)", df, df["mfi5a_cat"]),
        ("Extended index (mFI-9 + BMI <22)", df, df["ext_cat"]),
    ]
    for name, mask in [("Excluding emergency admission", df["AD_STATUS_new"] != 0),
                       ("Aneurysm diameter >=50 mm", df["sac_size"] >= 50),
                       ("Excluding 2026 index repairs", df["TRT_yr_new"] < 2026)]:
        sub = df[mask].reset_index(drop=True)
        analyses.append((name, sub, sub["mFI9_cat3"]))

    rows, evals = [], []
    for name, d, s in analyses:
        r, e = run(name, d, s.reset_index(drop=True), n_boot, rng)
        rows += r
        evals.append(e)
        print(name, [f"{x['rd']:.1f}" for x in r], flush=True)

    # continuous mFI-9 slope
    est = slope(df)
    bs = []
    for _ in range(n_boot):
        try:
            b, _s = resample(df, df["mFI9_cat3"], rng)
            bs.append(slope(b))
        except Exception:
            continue
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    cont = pd.DataFrame([dict(estimate=100 * est, lo=100 * lo, hi=100 * hi,
                              p=2 * norm.sf(abs(est) / bs.std(ddof=1)), n_boot=len(bs))])
    print("slope", cont.round(2).to_dict("records"))

    with pd.ExcelWriter(out) as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="Sens_RD", index=False)
        pd.DataFrame(evals).to_excel(xw, sheet_name="Evalue", index=False)
        cont.to_excel(xw, sheet_name="Continuous_mFI9", index=False)
        for name, d, s in analyses[1:3]:
            ct = pd.crosstab(s, d["osr"]).rename(columns={0: "EVAR", 1: "OSR"})
            ct.to_excel(xw, sheet_name=("N_" + name[:10]).replace("/", "").replace(">", ""))


if __name__ == "__main__":
    main()
