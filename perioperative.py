"""Perioperative composite outcome: EVAR vs OSR within mFI-9 frailty strata.

Usage: python perioperative.py <dataset.xlsx> <out.xlsx> [n_boot]

Estimators (risk difference RD = OSR minus EVAR, in each mFI-9 stratum 0-1 / 2 / >=3):
  primary  stratum-specific propensity score (20 covariates, ow_core.COVARIATES) with
           overlap weights
  (a) IPTW stabilized inverse probability of treatment weights within strata
           (stratum-specific PS, weights P(T)/ps for OSR and (1-P(T))/(1-ps) for EVAR)
  (b) pooled PS a single propensity score on the 20 covariates fitted in the whole cohort
           (no stratum term; the mFI-9 items hypertension, diabetes, PTA, heart failure and
           chronic lung disease are already covariates) with overlap weights; RDs are then
           read within strata
  (c) additive interaction on the primary analysis: RD differences vs the robust stratum
           (bootstrap percentile CI, P from the bootstrap SE); RERI, attributable proportion
           (AP) and synergy index (S) for the frail stratum with robust EVAR as the reference
           cell; global 2-df test = Wald test on osr x C(stratum) and trend 1-df test = Wald
           test on osr x stratum score (0/1/2) in a weighted linear probability model with
           HC0 sandwich variance
  (d) multiplicative interaction: ratio of odds ratios from the weighted risks (bootstrap
           CI), and a global 2-df Wald test on osr x C(stratum) in a weighted logistic model
           with HC0 sandwich variance
  (e) components: pulmonary complication (GC_PULMONARY_new) and prolonged ventilation
           (GC_VENT_new), RDs and global / trend tests as in (c)
Bootstrap: n_boot replicates (default 2000), resampling within stratum x treatment, every
propensity score re-estimated in each replicate; random seed 20261012.

Output: one sheet "Raw" (one row per estimate: est, lo, hi, p_bootse; test rows: est = P)
and a sheet "Summary" with the main results formatted as in the paper (percentage points).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from ow_core import load, ps_fit, weighted_rd

warnings.filterwarnings("ignore")
Y = "periop_composite"
COMP = {"pulmonary": "GC_PULMONARY_new", "ventilation": "GC_VENT_new"}
LABELS = ["Robust", "Pre-frail", "Frail"]
SEED = 20261012


# ---------- weighted models with HC0 sandwich ----------
def wls(X, y, w):
    XtW = X.T * w
    B = np.linalg.inv(XtW @ X)
    b = B @ XtW @ y
    M = X * (w * (y - X @ b))[:, None]
    return b, B @ (M.T @ M) @ B


def wlogit(X, y, w):
    b = np.zeros(X.shape[1])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-X @ b))
        step = np.linalg.solve((X.T * (w * mu * (1 - mu))) @ X, X.T @ (w * (y - mu)))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    mu = 1 / (1 + np.exp(-X @ b))
    B = np.linalg.inv((X.T * (w * mu * (1 - mu))) @ X)
    M = X * (w * (y - mu))[:, None]
    return b, B @ (M.T @ M) @ B


def wald(b, V, L):
    L = np.atleast_2d(np.asarray(L, float))
    c = L @ b
    return float(chi2.sf(c @ np.linalg.solve(L @ V @ L.T, c), L.shape[0]))


def tests(df, w, y):
    s = df["mFI9_cat3"].to_numpy(float)
    t = df["osr"].to_numpy(float)
    yv = df[y].to_numpy(float)
    one = np.ones(len(df))
    s1, s2 = (s == 1) * 1.0, (s == 2) * 1.0
    Xs = np.column_stack([one, t, s1, s2, t * s1, t * s2])
    b, V = wls(Xs, yv, w)
    g_add = wald(b, V, [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    bl, Vl = wls(np.column_stack([one, t, s, t * s]), yv, w)
    trend = wald(bl, Vl, [0, 0, 0, 1])
    out = dict(global_add_p=g_add, trend_p=trend)
    if y == Y:
        bg, Vg = wlogit(Xs, yv, w)
        out["global_mult_p"] = wald(bg, Vg, [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        out["ror_wald_p_frail"] = wald(bg, Vg, [0, 0, 0, 0, 0, 1])
    return out


# ---------- one pass of all estimators ----------
def odds(p):
    return p / (1 - p)


def estimate(df):
    s = df["mFI9_cat3"].to_numpy()
    t = df["osr"].to_numpy() == 1
    w_ow = np.empty(len(df))
    w_ip = np.empty(len(df))
    for k in range(3):
        m = s == k
        ps = ps_fit(df[m])
        tk = t[m]
        p = tk.mean()
        w_ow[m] = np.where(tk, 1 - ps, ps)
        w_ip[m] = np.where(tk, p / ps, (1 - p) / (1 - ps))
    ps_all = ps_fit(df)
    w_pool = np.where(t, 1 - ps_all, ps_all)
    r = {}
    for tag, w in [("ow", w_ow), ("iptw", w_ip), ("pool", w_pool)]:
        for y in [Y] + (list(COMP.values()) if tag == "ow" else []):
            for k in range(3):
                m = s == k
                r0, r1, rd = weighted_rd(df[m], y, w[m])
                r[f"{tag}|{y}|{k}|r0"], r[f"{tag}|{y}|{k}|r1"], r[f"{tag}|{y}|{k}|rd"] = r0, r1, rd
    g = lambda k, x: r[f"ow|{Y}|{k}|{x}"]
    ref = g(0, "r0")
    rr10, rr01, rr11 = g(0, "r1") / ref, g(2, "r0") / ref, g(2, "r1") / ref
    r["reri"] = rr11 - rr10 - rr01 + 1
    r["ap"] = r["reri"] / rr11
    r["s"] = (rr11 - 1) / ((rr10 - 1) + (rr01 - 1))
    for k in range(3):
        r[f"or|{k}"] = odds(g(k, "r1")) / odds(g(k, "r0"))
    for k in (1, 2):
        r[f"ror|{k}"] = r[f"or|{k}"] / r["or|0"]
        r[f"logror|{k}"] = np.log(r[f"ror|{k}"])
    for tag in ("ow", "iptw", "pool"):
        for y in [Y] + (list(COMP.values()) if tag == "ow" else []):
            for k in (1, 2):
                r[f"{tag}|{y}|d{k}"] = r[f"{tag}|{y}|{k}|rd"] - r[f"{tag}|{y}|0|rd"]
    return r, w_ow


def resample(df, rng):
    idx = []
    for k in range(3):
        for tt in (0, 1):
            ii = np.flatnonzero(((df["mFI9_cat3"] == k) & (df["osr"] == tt)).to_numpy())
            idx.append(rng.choice(ii, size=len(ii), replace=True))
    return df.iloc[np.concatenate(idx)].reset_index(drop=True)


def analyse(df, n_boot, rng):
    est, w = estimate(df)
    est.update({f"test|{y}|{k}": v for y in [Y] + list(COMP.values()) for k, v in tests(df, w, y).items()})
    boots = []
    for _ in range(n_boot):
        try:
            boots.append(estimate(resample(df, rng))[0])
        except Exception:
            continue
    B = pd.DataFrame(boots)
    res = {}
    for key, val in est.items():
        if key.startswith("test|"):
            res[key] = dict(est=val)
            continue
        col = B[key].replace([np.inf, -np.inf], np.nan)
        lo, hi = np.nanpercentile(col, [2.5, 97.5])
        se = np.nanstd(col, ddof=1)
        res[key] = dict(est=val, lo=lo, hi=hi, p_bootse=2 * norm.sf(abs(val) / se))
    res["n_boot"] = dict(est=len(B))
    return res


# ---------- summary table ----------
def fmt(r, key, scale=100, ci=True, p=False, nd=1):
    x = r[key]
    s = f"{scale * x['est']:.{nd}f}"
    if ci and "lo" in x:
        s += f" ({scale * x['lo']:.{nd}f} to {scale * x['hi']:.{nd}f})"
    if p and "p_bootse" in x:
        s += f"; P={x['p_bootse']:.3f}"
    return s


def summary(r):
    pul, ven = COMP["pulmonary"], COMP["ventilation"]
    rds = lambda tag, y=Y: "; ".join(fmt(r, f"{tag}|{y}|{k}|rd") for k in range(3))
    tp = lambda y, k: f"{r[f'test|{y}|{k}']['est']:.3f}"
    spec = [
        ("Primary OW RD robust / pre-frail / frail", rds("ow")),
        ("Primary: pre-frail minus robust", fmt(r, f"ow|{Y}|d1", p=True)),
        ("Primary: frail minus robust", fmt(r, f"ow|{Y}|d2", p=True)),
        ("(a) IPTW RD robust / pre-frail / frail", rds("iptw")),
        ("(a) IPTW: frail minus robust", fmt(r, f"iptw|{Y}|d2", p=True)),
        ("(b) Pooled PS RD robust / pre-frail / frail", rds("pool")),
        ("(b) Pooled PS: frail minus robust", fmt(r, f"pool|{Y}|d2", p=True)),
        ("(c) RERI frail", fmt(r, "reri", scale=1, nd=2)),
        ("(c) AP frail", fmt(r, "ap", scale=1, nd=2)),
        ("(c) S frail", fmt(r, "s", scale=1, nd=2)),
        ("(c) Global additive P (2-df Wald, WLS HC0)", tp(Y, "global_add_p")),
        ("(c) Trend P (osr x score, WLS HC0)", tp(Y, "trend_p")),
        ("(d) Ratio of ORs frail vs robust", fmt(r, "ror|2", scale=1, nd=2)
         + f"; P(boot SE, log)={r['logror|2']['p_bootse']:.3f}; P(logit Wald)={tp(Y, 'ror_wald_p_frail')}"),
        ("(d) Global multiplicative P (logit HC0)", tp(Y, "global_mult_p")),
        ("(e) Pulmonary RD robust / pre-frail / frail", rds("ow", pul)),
        ("(e) Pulmonary trend / global P", f"{tp(pul, 'trend_p')} / {tp(pul, 'global_add_p')}"),
        ("(e) Ventilation RD robust / pre-frail / frail", rds("ow", ven)),
        ("(e) Ventilation trend / global P", f"{tp(ven, 'trend_p')} / {tp(ven, 'global_add_p')}"),
        ("Bootstrap replicates", str(r["n_boot"]["est"])),
    ]
    return pd.DataFrame(spec, columns=["item", "value"])


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    df = load(src).reset_index(drop=True)
    res = analyse(df, n_boot, np.random.default_rng(SEED))
    print("n =", len(df), "| bootstrap replicates:", res["n_boot"]["est"], flush=True)
    tab = summary(res)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(tab.to_string(index=False))
    with pd.ExcelWriter(out) as xw:
        tab.to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(res).T.to_excel(xw, sheet_name="Raw")


if __name__ == "__main__":
    main()
