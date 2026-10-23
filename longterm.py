"""Long-term outcomes: EVAR vs OSR within mFI-9 frailty strata.

Usage: python longterm.py <dataset.xlsx> <out.xlsx> [n_boot]

Methods:
  - long-term set: FU_dur_yr_new > 0 (time FU_dur_yr_new in years, event All_cause_death_new)
  - overlap weights from a stratum-specific propensity score re-estimated on the long-term
    set (ow_core covariates); stabilized IPTW as the sensitivity weight
  - weighted Cox model (lifelines, Efron ties) with robust variance, per mFI-9 stratum
  - interaction: pooled weighted Cox with evar + stratum dummies + evar x dummies, 2-df
    robust Wald test; trend: evar + stratum score + evar x score
  - proportional hazards: lifelines proportional_hazard_test, time_transform='km'
  - piecewise HR: 0-2 years (administrative censoring at 2 years) and > 2 years (left
    truncation, risk set at 2 years)
  - 5-year RMST and 5-year mortality from the weighted Kaplan-Meier estimator; percentile
    bootstrap within stratum x treatment with the propensity score re-estimated in every
    replicate (random seed 20261022); P from the bootstrap standard error
  - reintervention: weighted Aalen-Johansen cumulative incidence at 5 years (t_prim,
    st_prim: 1 reintervention, 2 death as competing event), computed with exact ties
  - prognostic Cox model (unweighted, long-term set): frailty index count + AGE +
    GENDER_new + osr, for mFI-9, mFI-5a and two extended indices
Also reports median follow-up, aneurysm-related deaths, reintervention types and the
weighted Kaplan-Meier survival at 1, 3 and 5 years.

Output: <out.xlsx> (sheets Strata, Prognostic, Summary) and a text log <out.txt>.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy import stats

from ow_core import STRATA, load, ow_weights, ps_fit

warnings.filterwarnings("ignore")
T, E = "FU_dur_yr_new", "All_cause_death_new"
SN = ["Robust", "Pre-frail", "Frail"]


# ---------- weighted estimators (numpy, exact ties) ----------
def wkm(t, e, w):
    """Weighted KM: returns event times and survival just after each."""
    ut, inv = np.unique(t, return_inverse=True)
    d = np.bincount(inv, weights=w * (e == 1), minlength=len(ut))
    tot = np.bincount(inv, weights=w, minlength=len(ut))
    r = tot[::-1].cumsum()[::-1]                   # weighted risk set at each distinct time
    keep = d > 0
    return ut[keep], np.cumprod(1 - d[keep] / r[keep])


def surv_at(ut, s, x):
    i = np.searchsorted(ut, x, side="right") - 1
    return 1.0 if i < 0 else s[i]


def rmst(ut, s, tau=5.0):
    pts = np.concatenate([[0.0], ut[ut < tau], [tau]])
    vals = np.concatenate([[1.0], s[ut < tau]])
    return float(np.sum(np.diff(pts) * vals))


def waj(t, st, w, tau=5.0, k=1):
    """Weighted Aalen-Johansen cumulative incidence of cause k at tau."""
    o = np.argsort(t)
    t, st, w = t[o], st[o], w[o]
    s, cif = 1.0, 0.0
    for u in np.unique(t[(st > 0) & (t <= tau)]):
        r = w[t >= u].sum()
        dk = w[(t == u) & (st == k)].sum()
        da = w[(t == u) & (st > 0)].sum()
        cif += s * dk / r
        s *= 1 - da / r
    return cif


# ---------- helpers ----------
def add_weights(d):
    d = d.copy()
    d["w"] = np.nan
    d["w_iptw"] = np.nan
    for s in range(3):
        m = d["mFI9_cat3"] == s
        ds = d[m]
        d.loc[m, "w"] = ow_weights(ds)
        ps = ps_fit(ds)
        p1 = ds["osr"].mean()
        d.loc[m, "w_iptw"] = np.where(ds["osr"] == 1, p1 / ps, (1 - p1) / (1 - ps))
    return d


def cox(d, cols, w="w", **kw):
    return CoxPHFitter().fit(d[[T, E, w] + cols], T, E, weights_col=w, robust=True, **kw)


def hr_row(c, v="evar"):
    lo, hi = np.exp(c.confidence_intervals_.loc[v])
    return np.exp(c.params_[v]), lo, hi, c.summary.p[v]


def wald(c, names):
    b = c.params_[names].values
    V = c.variance_matrix_.loc[names, names].values
    return stats.chi2.sf(float(b @ np.linalg.solve(V, b)), len(names))


def fmt(h, lo, hi):
    return f"{h:.2f} ({lo:.2f}-{hi:.2f})"


def stratum_point(d):
    """RMST (years), 5-y mortality and 5-y reintervention per arm for one stratum (weights in d.w)."""
    r = {}
    for g in (1, 0):
        x = d[d["evar"] == g]
        ut, s = wkm(x[T].to_numpy(), x[E].to_numpy(), x["w"].to_numpy())
        r[g] = dict(rmst=rmst(ut, s), mort5=1 - surv_at(ut, s, 5.0),
                    km1=surv_at(ut, s, 1.0), km3=surv_at(ut, s, 3.0), km5=surv_at(ut, s, 5.0))
        xr = x[x["t_prim"].notna()]
        r[g]["aj5"] = waj(xr["t_prim"].to_numpy(), xr["st_prim"].to_numpy(), xr["w"].to_numpy())
    return r


def boot(lt, n_boot, seed=20261022):
    rng = np.random.default_rng(seed)
    res = {s: [] for s in range(3)}
    for _ in range(n_boot):
        for s in range(3):
            d = lt[lt["mFI9_cat3"] == s]
            idx = np.concatenate([rng.choice(np.flatnonzero(d["evar"].to_numpy() == g),
                                             (d["evar"] == g).sum(), replace=True) for g in (0, 1)])
            b = d.iloc[idx].copy()
            b["w"] = ow_weights(b)
            r = stratum_point(b)
            res[s].append((12 * (r[1]["rmst"] - r[0]["rmst"]), 100 * (r[1]["mort5"] - r[0]["mort5"])))
    return {s: np.array(v) for s, v in res.items()}


# ---------- main analysis ----------
def analyse(df, label, n_boot):
    lt = add_weights(df[df[T] > 0])
    out = {"label": label, "n_primary": len(df), "n_lt": len(lt), "deaths": int(lt[E].sum())}
    out["medfu"] = (lt[T].median(), lt.loc[lt.evar == 1, T].median(), lt.loc[lt.evar == 0, T].median())
    out["aneur"] = (int(lt.loc[lt.evar == 1, "aneurymal_related_death_new"].sum()),
                    int(lt.loc[lt.evar == 0, "aneurymal_related_death_new"].sum()))
    rp = df[df["st_prim"] == 1]
    out["reint"] = {g: rp.loc[rp.evar == g, "reint_type"].value_counts().to_dict() for g in (1, 0)}
    rows = []
    for s in range(3):
        d = lt[lt["mFI9_cat3"] == s]
        c = cox(d, ["evar"])
        ph = proportional_hazard_test(c, d[[T, E, "w", "evar"]], time_transform="km").summary.p.iloc[0]
        ci = cox(d, ["evar"], w="w_iptw")
        a = d.copy()
        a[T] = a[T].clip(upper=2.0)
        a[E] = np.where(d[T] <= 2.0, d[E], 0)
        p1 = hr_row(cox(a, ["evar"]))
        b = d[d[T] > 2.0].copy()
        b["entry"] = 2.0
        p2 = hr_row(CoxPHFitter().fit(b[[T, E, "w", "evar", "entry"]], T, E, weights_col="w",
                                      robust=True, entry_col="entry"))
        r = stratum_point(d)
        rows.append(dict(stratum=SN[s], n=len(d),
                         n_evar=int((d.evar == 1).sum()), n_osr=int((d.evar == 0).sum()),
                         deaths_evar=int(d.loc[d.evar == 1, E].sum()), deaths_osr=int(d.loc[d.evar == 0, E].sum()),
                         **dict(zip(["hr", "hr_lo", "hr_hi", "hr_p"], hr_row(c))), ph_p=ph,
                         **dict(zip(["hr_iptw", "iptw_lo", "iptw_hi", "iptw_p"], hr_row(ci))),
                         **dict(zip(["pw02", "pw02_lo", "pw02_hi", "pw02_p"], p1)),
                         **dict(zip(["pw2p", "pw2p_lo", "pw2p_hi", "pw2p_p"], p2)),
                         rmst_evar=r[1]["rmst"], rmst_osr=r[0]["rmst"],
                         rmst_diff_mo=12 * (r[1]["rmst"] - r[0]["rmst"]),
                         mort5_evar=100 * r[1]["mort5"], mort5_osr=100 * r[0]["mort5"],
                         mort5_diff=100 * (r[1]["mort5"] - r[0]["mort5"]),
                         aj5_evar=100 * r[1]["aj5"], aj5_osr=100 * r[0]["aj5"],
                         km1_evar=100 * r[1]["km1"], km3_evar=100 * r[1]["km3"], km5_evar=100 * r[1]["km5"],
                         km1_osr=100 * r[0]["km1"], km3_osr=100 * r[0]["km3"], km5_osr=100 * r[0]["km5"]))
    lt["s1"] = (lt.mFI9_cat3 == 1).astype(int)
    lt["s2"] = (lt.mFI9_cat3 == 2).astype(int)
    lt["e1"], lt["e2"] = lt.evar * lt.s1, lt.evar * lt.s2
    lt["sn"] = lt.mFI9_cat3.astype(float)
    lt["en"] = lt.evar * lt.sn
    out["p_int"] = wald(cox(lt, ["evar", "s1", "s2", "e1", "e2"]), ["e1", "e2"])
    out["p_trend"] = cox(lt, ["evar", "sn", "en"]).summary.p["en"]
    if n_boot:
        bs = boot(lt, n_boot)
        for s in range(3):
            v = bs[s]
            for j, k in enumerate(["rmst_diff_mo", "mort5_diff"]):
                est = rows[s][k]
                se = v[:, j].std(ddof=1)
                rows[s][k + "_lo"], rows[s][k + "_hi"] = np.percentile(v[:, j], [2.5, 97.5])
                rows[s][k + "_p"] = 2 * stats.norm.sf(abs(est / se))
        out["n_boot"] = n_boot
    out["strata"] = pd.DataFrame(rows)
    # prognostic Cox (unweighted)
    p = lt.copy()
    p["ext_bmi"] = p.mFI9_count + (p.BMI < 22).astype(int)
    p["ext_bmi_er"] = p.ext_bmi + (p.AD_STATUS_new == 0).astype(int)
    pr = []
    for idx in ["mFI9_count", "mFI5a_count", "ext_bmi", "ext_bmi_er"]:
        c = CoxPHFitter().fit(p[[T, E, idx, "AGE", "GENDER_new", "osr"]], T, E)
        lo, hi = np.exp(c.confidence_intervals_.loc[idx])
        pr.append(dict(index=idx, hr=np.exp(c.params_[idx]), lo=lo, hi=hi, p=c.summary.p[idx],
                       aic=c.AIC_partial_, concordance=c.concordance_index_))
    out["prog"] = pd.DataFrame(pr)
    return out, lt


def main():
    src, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    r, _ = analyse(load(src), "cohort", n_boot)
    st = r["strata"]
    log = [f"primary n={r['n_primary']}, long-term n={r['n_lt']}, deaths={r['deaths']}"]
    log.append("median FU overall/EVAR/OSR %.2f / %.2f / %.2f y; aneurysm-related deaths EVAR/OSR %s"
               % (*r["medfu"], r["aneur"]))
    log.append(f"reintervention types EVAR {r['reint'][1]}  OSR {r['reint'][0]}")
    log.append("P interaction 2-df %.3f, trend %.3f" % (r["p_int"], r["p_trend"]))
    for _, x in st.iterrows():
        s = (f"  {x.stratum:9s} n={x.n} deaths {x.deaths_evar}/{x.deaths_osr} HR {fmt(x.hr, x.hr_lo, x.hr_hi)} "
             f"P={x.hr_p:.3f} PH P={x.ph_p:.3f} IPTW {fmt(x.hr_iptw, x.iptw_lo, x.iptw_hi)} | "
             f"pw 0-2 {fmt(x.pw02, x.pw02_lo, x.pw02_hi)} >2 {fmt(x.pw2p, x.pw2p_lo, x.pw2p_hi)} | "
             f"RMST {x.rmst_evar:.2f}/{x.rmst_osr:.2f} diff {x.rmst_diff_mo:.1f} mo")
        if "rmst_diff_mo_lo" in st:
            s += (f" ({x.rmst_diff_mo_lo:.1f} to {x.rmst_diff_mo_hi:.1f}; P={x.rmst_diff_mo_p:.3f})"
                  f" | 5y mort {x.mort5_evar:.1f}/{x.mort5_osr:.1f} diff {x.mort5_diff:.1f}"
                  f" ({x.mort5_diff_lo:.1f} to {x.mort5_diff_hi:.1f})")
        s += (f" | AJ5 {x.aj5_evar:.1f}/{x.aj5_osr:.1f} | KM1/3/5 EVAR {x.km1_evar:.1f}/{x.km3_evar:.1f}/"
              f"{x.km5_evar:.1f} OSR {x.km1_osr:.1f}/{x.km3_osr:.1f}/{x.km5_osr:.1f}")
        log.append(s)
    for _, x in r["prog"].iterrows():
        log.append(f"  prognostic {x['index']:12s} HR/item {fmt(x.hr, x.lo, x.hi)} P={x.p:.4f} "
                   f"C={x.concordance:.3f} AIC={x.aic:.1f}")
    txt = "\n".join(log)
    print(txt)
    out_p.with_suffix(".txt").write_text(txt, encoding="utf-8")
    with pd.ExcelWriter(out_p) as xw:
        st.to_excel(xw, sheet_name="Strata", index=False)
        r["prog"].to_excel(xw, sheet_name="Prognostic", index=False)
        pd.DataFrame([dict(n_primary=r["n_primary"], n_lt=r["n_lt"], deaths=r["deaths"],
                           medfu_all=r["medfu"][0], medfu_evar=r["medfu"][1], medfu_osr=r["medfu"][2],
                           aneur_evar=r["aneur"][0], aneur_osr=r["aneur"][1],
                           reint_evar=str(r["reint"][1]), reint_osr=str(r["reint"][0]),
                           p_int=r["p_int"], p_trend=r["p_trend"], n_boot=r.get("n_boot", 0))]
                     ).to_excel(xw, sheet_name="Summary", index=False)


if __name__ == "__main__":
    main()
