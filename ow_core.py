"""Overlap-weighting core shared by all analysis scripts.

Stratum-specific logistic propensity score (OSR vs EVAR) on 20 pre-operative
covariates, overlap weights (OSR: 1 - ps, EVAR: ps), weighted risks and risk
differences.
"""
from pathlib import Path

import numpy as np
import pandas as pd
# GLM is imported directly from statsmodels.genmod to avoid loading statsmodels.api
# (and its compiled time-series modules), which is not needed here.
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools import add_constant

COVARIATES = [
    "AGE", "GENDER_new", "BMI", "SMOKING_new", "DYSLIPIDEMIA_new", "cad",
    "PREV_CAD_new", "PREV_CVA_new", "PREV_RF_new", "ECG_FIBRIL_new",
    "sac_size", "cia30", "hostile_neck", "TRT_yr_G_new", "AD_STATUS_new",
    "HYPERTENSION_new", "DIABETES_new", "PREV_PTA_new", "PREV_HF_new",
    "PREV_CLD_new",
]
STRATA = {0: "Robust (0-1)", 1: "Pre-frail (2)", 2: "Frail (>=3)"}


def load(dataset: Path) -> pd.DataFrame:
    df = pd.read_excel(dataset, sheet_name="Patient_level")
    df = df[df["analysis_set_frailty"] == 1].copy()
    df["osr"] = 1 - df["evar"]
    df["cia30"] = df["cia30"].fillna(0)
    return df


def ps_fit(d: pd.DataFrame, covs=COVARIATES) -> np.ndarray:
    X = d[covs].astype(float)
    X = X.loc[:, X.std() > 0]                      # drop constants within a stratum
    X = add_constant(X, has_constant="add")
    m = GLM(d["osr"].astype(float), X, family=families.Binomial()).fit()
    return np.asarray(m.predict(X))


def ow_weights(d: pd.DataFrame, covs=COVARIATES) -> np.ndarray:
    ps = ps_fit(d, covs)
    return np.where(d["osr"] == 1, 1 - ps, ps)


def weighted_rd(d: pd.DataFrame, y: str, w: np.ndarray):
    """Weighted risk in EVAR (r0), in OSR (r1) and the risk difference r1 - r0."""
    t = d["osr"].to_numpy() == 1
    yv = d[y].to_numpy(dtype=float)
    r1 = np.average(yv[t], weights=w[t])
    r0 = np.average(yv[~t], weights=w[~t])
    return r0, r1, r1 - r0
