# aaa-frailty-evar-osr

Analysis code for the paper

> **Frailty and the absolute perioperative benefit of endovascular over open repair of intact
> abdominal aortic aneurysm: a propensity-weighted cohort study** (submitted to *Scientific
> Reports*).

The cohort comprises 1,697 patients with intact abdominal aortic aneurysm treated by
endovascular aneurysm repair (EVAR) or open surgical repair (OSR) in the Asan Medical Center
VAPOR registry. Patients are stratified by the nine-item modified frailty index (mFI-9:
robust 0-1, pre-frail 2, frail >=3 points), and within each stratum EVAR and OSR are
compared with overlap weights from a stratum-specific propensity score.

**Patient data are not included in this repository.** See [Data availability](#data-availability).

## Files

| File | Content |
|------|---------|
| `ow_core.py` | Shared core: data loading, the 20 propensity-score covariates, stratum-specific logistic propensity score, overlap weights, weighted risks and risk differences |
| `perioperative.py` | Perioperative composite outcome: primary overlap-weighted risk differences (RD) per stratum; stabilized IPTW and pooled-propensity-score sensitivity estimators; additive interaction (RD differences, RERI, AP, S, global and trend Wald tests); multiplicative interaction (ratio of odds ratios, global Wald test); components (pulmonary complication, mechanical ventilation) |
| `longterm.py` | Long-term outcomes: weighted Cox hazard ratios per stratum (overlap weights, IPTW), interaction and trend tests, proportional-hazards test, piecewise HRs (0-2 and >2 years), 5-year RMST and mortality differences (bootstrap), weighted Aalen-Johansen reintervention incidence, prognostic Cox models for each frailty index |
| `sensitivity_indices.py` | Sensitivity analyses I: mFI-5a and an extended index (mFI-9 + BMI < 22) as stratifiers; restrictions (excluding emergency admissions, diameter >= 50 mm, excluding 2026 repairs); E-values; continuous mFI-9 modifier |
| `sensitivity_models.py` | Sensitivity analyses II: severe composite outcome; restricted cubic splines for continuous covariates; treatment era as calendar year or three periods; mFI-5a in two strata; propensity-score diagnostics (variance ratios, effective sample size, c statistic); reintervention definitions |

Risk differences are always OSR minus EVAR, in percentage points (positive = excess risk
with OSR). The propensity score is fitted as the probability of OSR; overlap weights are
symmetric in the treatment coding, so this is equivalent to modelling the probability of
EVAR.

## Requirements

Python 3.14 (developed and tested with 3.14.3). Packages, with the versions used, are in
`requirements.txt`:

```
pip install -r requirements.txt
```

`statsmodels` is used only through `statsmodels.genmod` (GLM) and `statsmodels.tools`;
`statsmodels.api` is never imported.

## Usage

All scripts take the dataset (an Excel workbook with a `Patient_level` sheet, see below) and
an output workbook path; the optional last argument is the number of bootstrap replicates
(default 2000). Run them from this directory so that `ow_core.py` can be imported.

```
python perioperative.py        <dataset.xlsx> <out.xlsx> [n_boot]
python longterm.py             <dataset.xlsx> <out.xlsx> [n_boot]
python sensitivity_indices.py  <dataset.xlsx> <out.xlsx> [n_boot]
python sensitivity_models.py   <dataset.xlsx|patient_level.pkl> <out.xlsx> [n_boot]
```

| Script | Output |
|--------|--------|
| `perioperative.py` | sheets `Summary` (formatted main results) and `Raw` (every estimate with bootstrap 2.5/97.5 percentiles and P from the bootstrap SE; test rows hold the Wald P in `est`) |
| `longterm.py` | sheets `Strata`, `Prognostic`, `Summary`, plus a text log `<out>.txt` |
| `sensitivity_indices.py` | sheets `Sens_RD`, `Evalue`, `Continuous_mFI9` and stratum-by-treatment counts for the alternative indices |
| `sensitivity_models.py` | sheets `Sens_RD`, `Evalue`, `PS_diagnostics`, `PS_values`, `Reintervention`, `Severe_counts`; also accepts a pickled `Patient_level` DataFrame |

With 2000 replicates each script takes from several minutes to about an hour on a laptop;
`n_boot = 20` is enough to check that the code runs.

### Bootstrap

Percentile bootstrap, resampling with replacement within each stratum x treatment cell; the
propensity score (and hence the weights) is re-estimated in every replicate. P values for
differences use the bootstrap standard error (normal approximation). Fixed seeds
(`numpy.random.default_rng`):

| Script | Seed |
|--------|------|
| `perioperative.py` | 20261012 |
| `longterm.py` | 20261022 |
| `sensitivity_indices.py` | 20261012 (one generator used sequentially across analyses) |
| `sensitivity_models.py` | 20261014 (one generator used sequentially across analyses) |

Point estimates do not depend on the seed; confidence limits and bootstrap P values vary
slightly with the seed and the number of replicates.

## Data dictionary

The scripts read the sheet `Patient_level` (one row per patient) and keep rows with
`analysis_set_frailty == 1`. Binary variables are coded 0 = no, 1 = yes unless stated.
Only the columns below are used.

### Analysis set, treatment and frailty

| Column | Type | Description |
|--------|------|-------------|
| `analysis_set_frailty` | 0/1 | 1 = included in the analysis cohort (n = 1,697) |
| `evar` | 0/1 | Repair type: 1 = EVAR, 0 = OSR (the scripts derive `osr = 1 - evar`) |
| `mFI9_count` | int 0-9 | Number of mFI-9 items present: hypertension, diabetes, coronary bypass, myocardial infarction, percutaneous coronary intervention, peripheral angioplasty, heart failure, cerebrovascular disease, chronic lung disease |
| `mFI9_cat3` | 0/1/2 | mFI-9 stratum: 0 = robust (0-1), 1 = pre-frail (2), 2 = frail (>=3) |
| `mFI5a_count` | int 0-4 | mFI-5a count: hypertension, diabetes, heart failure, chronic lung disease |

### Propensity-score covariates (`ow_core.COVARIATES`)

| Column | Type | Description |
|--------|------|-------------|
| `AGE` | years | Age at repair |
| `GENDER_new` | 0/1 | Sex: 0 = female, 1 = male |
| `BMI` | kg/m2 | Body-mass index |
| `SMOKING_new` | 0/1 | Ever smoking |
| `DYSLIPIDEMIA_new` | 0/1 | Dyslipidaemia |
| `cad` | 0/1 | Coronary artery disease (myocardial infarction, percutaneous coronary intervention or coronary bypass) |
| `PREV_CAD_new` | 0/1 | Carotid artery disease |
| `PREV_CVA_new` | 0/1 | Cerebrovascular disease |
| `PREV_RF_new` | 0/1 | History of renal failure |
| `ECG_FIBRIL_new` | 0/1 | Atrial fibrillation |
| `sac_size` | mm | Maximum aneurysm diameter |
| `cia30` | 0/1 | Common iliac artery diameter >= 30 mm (missing set to 0) |
| `hostile_neck` | 0/1 | Hostile proximal neck (short, angulated, conical or wide) |
| `TRT_yr_G_new` | 0/1 | Treatment era: 1 = repair in 2017 or later |
| `AD_STATUS_new` | 0/1 | Admission route: 0 = emergency department, 1 = outpatient clinic |
| `HYPERTENSION_new` | 0/1 | Hypertension (mFI item) |
| `DIABETES_new` | 0/1 | Diabetes mellitus (mFI item) |
| `PREV_PTA_new` | 0/1 | Peripheral angioplasty (mFI item) |
| `PREV_HF_new` | 0/1 | Heart failure (mFI item) |
| `PREV_CLD_new` | 0/1 | Chronic lung disease (mFI item) |

Covariates that are constant within a stratum (or a bootstrap replicate) are dropped from
that stratum's propensity model.

### Perioperative outcomes

| Column | Type | Description |
|--------|------|-------------|
| `periop_composite` | 0/1 | Primary outcome: any general complication, in-hospital MACE (death, myocardial infarction, stroke, major bleeding) or death in hospital or within 30 days |
| `GC_PULMONARY_new` | 0/1 | Pulmonary complication |
| `GC_VENT_new` | 0/1 | Mechanical ventilation |
| `death_inhosp_or_30d` | 0/1 | Death in hospital or within 30 days (severe composite) |
| `CE_MI_new` | 0/1 | In-hospital myocardial infarction (severe composite) |
| `CE_CVA_new` | 0/1 | In-hospital stroke (severe composite) |
| `CE_BL_new` | 0/1 | In-hospital major bleeding (severe composite) |
| `GC_RENAL_FAIL_RD_new` | 0/1 | Renal failure requiring dialysis (severe composite) |

### Long-term outcomes and other variables

| Column | Type | Description |
|--------|------|-------------|
| `FU_dur_yr_new` | years | Follow-up time from the index repair to death or last follow-up; the long-term set is `FU_dur_yr_new > 0` |
| `All_cause_death_new` | 0/1 | All-cause death |
| `aneurymal_related_death_new` | 0/1 | Aneurysm-related death (column name spelled as in the registry export) |
| `t_prim` | years | Time to first aortic reintervention, death or censoring |
| `st_prim` | 0/1/2 | Status at `t_prim`: 0 = censored, 1 = reintervention, 2 = death without reintervention |
| `reint_type` | text | Type of first reintervention: `Additional EVT` or `Additional Surgery` (empty if none) |
| `Reint_any` | 0/1 | Any reintervention record, including those not counted in the outcome definition |
| `Reint_same_day_as_index` | 0/1 | Reintervention record dated on the day of the index repair |
| `TRT_yr_new` | year | Calendar year of the index repair (era sensitivity analyses, 2026 exclusion) |

Derived in the scripts: `osr`; the extended index `mFI9_count + (BMI < 22)`; the severe
composite; spline bases for `AGE`, `BMI`, `sac_size` (knots at the 5th, 35th, 65th and 95th
percentiles); era periods 2011-2015, 2016-2020 and 2021-2026.

## Data availability

The individual patient data are not publicly available: they form part of an institutional
AAA registry at Asan Medical Center and contain potentially identifying clinical
information, restricted under Korean personal-data-protection regulations and the IRB
approval. De-identified data may be shared with qualified researchers on reasonable request
to the corresponding author, subject to approval by Asan Medical Center and a data-use
agreement, as stated in the paper's Data availability section. A dataset with the columns
listed above, in a `Patient_level` sheet, is all the scripts need.

## License

MIT; see `LICENSE`.
