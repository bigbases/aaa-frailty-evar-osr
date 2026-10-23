"""Sensitivity analyses II: outcome definition and propensity-model specification.

Usage: python sensitivity_models.py <dataset.xlsx|patient_level.pkl> <out.xlsx> [n_boot]

Analyses (all repeat the primary method: stratum-specific propensity score, overlap
weights, percentile bootstrap within treatment x stratum, propensity score re-estimated
in every replicate):
  - severe composite: in-hospital or 30-day death, MI, stroke, major bleeding and renal
    failure requiring dialysis (no general complication, no ventilation)
  - restricted cubic splines (4 knots) for age, body-mass index and aneurysm diameter
  - treatment era as calendar year (linear) and as three periods
  - mFI-5a in two strata (0-1 vs >=2)
Also writes propensity-score diagnostics (variance ratios, ESS, overlap) and the
reintervention definitions/types. Random seed 20261014 (one generator shared
sequentially by all analyses). A .pkl input must hold the Patient_level table.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ow_core import COVARIATES, load, ow_weights, ps_fit, weighted_rd

warnings.filterwarnings("ignore")
Y = "periop_composite"
YS = "severe_composite"


def rcs(x, knots):
    """Harrell restricted cubic spline basis (non-linear terms only)."""
    k = len(knots)
    t = np.asarray(knots, float)
    p = lambda u: np.clip(u, 0, None) ** 3
    cols = []
    for j in range(k - 2):
        c = (p(x - t[j]) - p(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
             + p(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(c / (t[k - 1] - t[0]) ** 2)
    return np.column_stack(cols)


def strata_rd(df, strat, K, y, covs):
    out = []
    for k in range(K):
        d = df[(strat == k).to_numpy()]
        out.append(weighted_rd(d, y, ow_weights(d, covs)))
    return out


def resample(df, strat, K, rng):
    idx = []
    for k in range(K):
        for t in (0, 1):
            ii = np.flatnonzero(((strat == k) & (df["osr"] == t)).to_numpy())
            idx.append(rng.choice(ii, size=len(ii), replace=True))
    idx = np.concatenate(idx)
    return df.iloc[idx].reset_index(drop=True), strat.iloc[idx].reset_index(drop=True)


def evalue(rr):
    rr = 1 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1)) if rr > 1 else 1.0


def run(name, df, strat, labels, n_boot, rng, y=Y, covs=COVARIATES):
    K = len(labels)
    strat = strat.reset_index(drop=True)
    df = df.reset_index(drop=True)
    est = strata_rd(df, strat, K, y, covs)
    boots = []
    for _ in range(n_boot):
        try:
            b, s = resample(df, strat, K, rng)
            r = strata_rd(b, s, K, y, covs)
            top = r[K - 1]
            boots.append([x[2] for x in r] + [top[1] / top[0] if top[0] > 0 else np.nan])
        except Exception:
            continue
    B = np.array(boots, dtype=float)
    rows = []
    for k in range(K):
        r0, r1, rd = est[k]
        lo, hi = np.nanpercentile(B[:, k], [2.5, 97.5])
        m = strat == k
        rows.append(dict(analysis=name, stratum=labels[k], n=int(m.sum()),
                         evar_events=f"{int((m & (df.osr == 0) & (df[y] == 1)).sum())}/{int((m & (df.osr == 0)).sum())}",
                         osr_events=f"{int((m & (df.osr == 1) & (df[y] == 1)).sum())}/{int((m & (df.osr == 1)).sum())}",
                         risk_evar=100 * r0, risk_osr=100 * r1, rd=100 * rd,
                         rd_lo=100 * lo, rd_hi=100 * hi))
    d = B[:, K - 1] - B[:, 0]
    delta = est[K - 1][2] - est[0][2]
    se = np.nanstd(d, ddof=1)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    rows.append(dict(analysis=name, stratum="Frail - robust", rd=100 * delta, rd_lo=100 * lo,
                     rd_hi=100 * hi, p=2 * norm.sf(abs(delta) / se), n_boot=len(B)))
    top = est[K - 1]
    rr = top[1] / top[0] if top[0] > 0 else np.nan
    rr_lo = np.nanpercentile(B[:, K], 2.5)
    ev = dict(analysis=name, rr_frail=rr, rr_lo=rr_lo,
              evalue_point=evalue(rr) if np.isfinite(rr) else np.nan,
              evalue_ci=evalue(rr_lo) if rr_lo > 1 else 1.0)
    print(name, [f"{x['rd']:.1f}" for x in rows], f"P={rows[-1]['p']:.3f}", flush=True)
    return rows, ev


def wvar(x, w):
    m = np.average(x, weights=w)
    return np.average((x - m) ** 2, weights=w)


def ps_diagnostics(df):
    rows, ps_rows = [], []
    for k, lab in enumerate(["Robust", "Pre-frail", "Frail"]):
        d = df[df["mFI9_cat3"] == k]
        ps = ps_fit(d)
        w = np.where(d["osr"] == 1, 1 - ps, ps)
        t = d["osr"].to_numpy() == 1
        for v in ["AGE", "BMI", "sac_size"]:
            x = d[v].to_numpy(float)
            rows.append(dict(stratum=lab, variable=v,
                             vr_unweighted=x[t].var(ddof=1) / x[~t].var(ddof=1),
                             vr_weighted=wvar(x[t], w[t]) / wvar(x[~t], w[~t])))
        ess = lambda ww: ww.sum() ** 2 / (ww ** 2).sum()
        # c statistic of the stratum propensity model
        o = np.argsort(ps)
        r = np.empty(len(ps)); r[o] = np.arange(1, len(ps) + 1)
        n1, n0 = t.sum(), (~t).sum()
        auc = (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        rows.append(dict(stratum=lab, variable="ESS EVAR / OSR",
                         vr_unweighted=float((~t).sum()), vr_weighted=ess(w[~t]),
                         note=f"OSR n={n1}, ESS={ess(w[t]):.0f}; c={auc:.3f}"))
        ps_rows += [dict(stratum=lab, osr=int(tt), ps=p) for tt, p in zip(t, ps)]
    return pd.DataFrame(rows), pd.DataFrame(ps_rows)


def reint_table(df):
    e = df[df["evar"] == 1]
    o = df[df["evar"] == 0]
    out = []
    for name, d in [("EVAR", e), ("OSR", o)]:
        out.append(dict(group=name, n=len(d), st_prim=int((d.st_prim == 1).sum()),
                        Reint_any=int(d.Reint_any.sum()),
                        any_not_prim=int(((d.Reint_any == 1) & (d.st_prim != 1)).sum()),
                        prim_not_any=int(((d.Reint_any == 0) & (d.st_prim == 1)).sum()),
                        any_not_prim_sameday=int(((d.Reint_any == 1) & (d.st_prim != 1)
                                                   & (d.Reint_same_day_as_index == 1)).sum()),
                        prim_type_EVT=int(((d.st_prim == 1) & (d.reint_type == "Additional EVT")).sum()),
                        prim_type_Surgery=int(((d.st_prim == 1) & (d.reint_type == "Additional Surgery")).sum()),
                        prim_type_missing=int(((d.st_prim == 1) & d.reint_type.isna()).sum())))
    return pd.DataFrame(out)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    rng = np.random.default_rng(20261014)
    if src.suffix == ".pkl":
        df = pd.read_pickle(src)
        df = df[df["analysis_set_frailty"] == 1].copy()
        df["osr"] = 1 - df["evar"]
        df["cia30"] = df["cia30"].fillna(0)
    else:
        df = load(src)
    df = df.reset_index(drop=True)

    df[YS] = ((df["death_inhosp_or_30d"] == 1) | (df["CE_MI_new"] == 1) | (df["CE_CVA_new"] == 1)
              | (df["CE_BL_new"] == 1) | (df["GC_RENAL_FAIL_RD_new"] == 1)).astype(int)
    # splines: knots at the 5/35/65/95th percentiles of the whole cohort
    spl = []
    for v in ["AGE", "BMI", "sac_size"]:
        kn = np.percentile(df[v], [5, 35, 65, 95])
        B = rcs(df[v].to_numpy(float), kn)
        for j in range(B.shape[1]):
            df[f"{v}_s{j}"] = B[:, j]
            spl.append(f"{v}_s{j}")
    yr = [c for c in COVARIATES if c != "TRT_yr_G_new"]
    df["era2"] = ((df["TRT_yr_new"] >= 2016) & (df["TRT_yr_new"] <= 2020)).astype(int)
    df["era3"] = (df["TRT_yr_new"] >= 2021).astype(int)
    df["mfi5a_2"] = (df["mFI5a_count"] >= 2).astype(int)

    L3 = ["Robust", "Pre-frail", "Frail"]
    s = df["mFI9_cat3"]
    analyses = [
        ("Primary (mFI-9)", df, s, L3, Y, COVARIATES),
        ("Severe composite", df, s, L3, YS, COVARIATES),
        ("Splines for continuous covariates", df, s, L3, Y, COVARIATES + spl),
        ("Era as calendar year", df, s, L3, Y, yr + ["TRT_yr_new"]),
        ("Era in three periods", df, s, L3, Y, yr + ["era2", "era3"]),
        ("mFI-5a (0-1 / >=2)", df, df["mfi5a_2"], ["Robust", "Frail"], Y, COVARIATES),
    ]
    rows, evals = [], []
    for name, d, st, lab, y, covs in analyses:
        r, e = run(name, d, st, lab, n_boot, rng, y, covs)
        rows += r
        evals.append(e)

    diag, ps = ps_diagnostics(df)
    print(diag.round(3).to_string(), flush=True)
    rt = reint_table(df)
    print(rt.to_string(), flush=True)
    sev = df.groupby(["mFI9_cat3", "osr"])[YS].agg(["sum", "count"])
    with pd.ExcelWriter(out) as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="Sens_RD", index=False)
        pd.DataFrame(evals).to_excel(xw, sheet_name="Evalue", index=False)
        diag.to_excel(xw, sheet_name="PS_diagnostics", index=False)
        ps.to_excel(xw, sheet_name="PS_values", index=False)
        rt.to_excel(xw, sheet_name="Reintervention", index=False)
        sev.to_excel(xw, sheet_name="Severe_counts")


if __name__ == "__main__":
    main()
