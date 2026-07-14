"""
Build a large-N survival-analysis dataset by merging multiple NHANES cycles
(1999-2018) with NCHS's public-use Linked Mortality Files.

Output columns:
    participant_id      - unique participant ID (SEQN)
    cycle               - NHANES cycle (e.g. "2007-2008")
    age                 - age in years at interview
    sex                 - "M" or "F", from RIAGENDR (see SEX_LABELS)
    race_eth            - race/ethnicity, from RIDRETH1 (see RACE_ETH_LABELS)
    education           - DMDEDUC2 category code (adults 20+)
    bmi                 - body mass index (kg/m^2)
    waist_cm            - waist circumference (cm)
    systolic_bp_mmhg    - mean systolic blood pressure across exam readings (BPXSY*, mm Hg)
    diastolic_bp_mmhg   - mean diastolic blood pressure across exam readings (BPXDI*, mm Hg)
    EVER_SMOKED         - smoked >=100 cigarettes in life, from SMQ020 (see SMOKED_LABELS)
    months_in_study     - person-months of follow-up from MEC exam date (PERMTH_EXM)
    event               - vital status at end of follow-up, from MORTSTAT
                          (see EVENT_LABELS: "deceased" or "alive")

Requires: pandas, requests  (pd.read_sas natively supports .XPT, no extra deps)
    pip install pandas requests

Run from the health_ml_tutorial directory; the output CSVs are saved to
examples/, where the gallery examples read them from:
    python download_nhanes.py
"""

import time
from io import BytesIO

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIG: continuous NHANES cycles and their file-name suffixes
# ---------------------------------------------------------------------------
CYCLES = [
    {"years": "1999-2000", "suffix": "",   "mort_years": "1999_2000"},
    {"years": "2001-2002", "suffix": "_B", "mort_years": "2001_2002"},
    {"years": "2003-2004", "suffix": "_C", "mort_years": "2003_2004"},
    {"years": "2005-2006", "suffix": "_D", "mort_years": "2005_2006"},
    {"years": "2007-2008", "suffix": "_E", "mort_years": "2007_2008"},
    {"years": "2009-2010", "suffix": "_F", "mort_years": "2009_2010"},
    {"years": "2011-2012", "suffix": "_G", "mort_years": "2011_2012"},
    {"years": "2013-2014", "suffix": "_H", "mort_years": "2013_2014"},
    {"years": "2015-2016", "suffix": "_I", "mort_years": "2015_2016"},
    {"years": "2017-2018", "suffix": "_J", "mort_years": "2017_2018"},
]

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
MORT_BASE = "https://ftp.cdc.gov/pub/health_statistics/nchs/datalinkage/linked_mortality"

# RIAGENDR codes, from the NHANES DEMO data-file codebook
SEX_LABELS = {
    1: "M",
    2: "F",
}

# RIDRETH1 codes, from the NHANES DEMO data-file codebook
RACE_ETH_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    5: "Other race - including multi-racial",
}

# SMQ020 codes, from the NHANES SMQ data-file codebook
SMOKED_LABELS = {
    1: "yes",
    2: "no",
    7: "refused",
    9: "don't know",
}

# MORTSTAT codes, from the NCHS Linked Mortality File codebook
EVENT_LABELS = {
    0: "alive",
    1: "deceased",
}

HORIZON_MONTHS = 5 * 12  # fixed 5-year mortality horizon

# Fixed-width column spec for the NCHS public-use mortality files
# (0-indexed, half-open intervals -- verified against the NCHS codebook)
MORT_COLSPECS = [
    (0, 6),    # SEQN
    (14, 15),  # ELIGSTAT      1=eligible, 2=under 18 (not released), 3=ineligible
    (15, 16),  # MORTSTAT      0=assumed alive, 1=assumed deceased
    (16, 19),  # UCOD_LEADING  leading cause-of-death group
    (19, 20),  # DIABETES      diabetes flag from cause-of-death
    (20, 21),  # HYPERTEN      hypertension flag from cause-of-death
    (42, 45),  # PERMTH_INT    person-months follow-up from interview date
    (45, 48),  # PERMTH_EXM    person-months follow-up from MEC/exam date
]
MORT_NAMES = [
    "SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
    "DIABETES", "HYPERTEN", "PERMTH_INT", "PERMTH_EXM",
]


def fetch_xpt(url: str) -> pd.DataFrame:
    """Download a SAS transport (.XPT) file and return it as a DataFrame."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_sas(BytesIO(r.content), format="xport")


def fetch_mortality(url: str) -> pd.DataFrame:
    """Download and parse a fixed-width NCHS linked mortality (.dat) file."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_fwf(BytesIO(r.content), colspecs=MORT_COLSPECS, names=MORT_NAMES)
    numeric_cols = ["MORTSTAT", "UCOD_LEADING", "DIABETES", "HYPERTEN",
                    "PERMTH_INT", "PERMTH_EXM"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def process_cycle(cycle: dict) -> pd.DataFrame | None:
    years, suffix, mort_years = cycle["years"], cycle["suffix"], cycle["mort_years"]
    print(f"Processing {years} ...")

    first_year = years.split("-")[0]
    files_base = f"{NHANES_BASE}/{first_year}/DataFiles"
    try:
        demo = fetch_xpt(f"{files_base}/DEMO{suffix}.XPT")
        bmx  = fetch_xpt(f"{files_base}/BMX{suffix}.XPT")
        bpx  = fetch_xpt(f"{files_base}/BPX{suffix}.XPT")
        smq  = fetch_xpt(f"{files_base}/SMQ{suffix}.XPT")
        mort = fetch_mortality(f"{MORT_BASE}/NHANES_{mort_years}_MORT_2019_PUBLIC.dat")
    except Exception as e:
        print(f"  Skipped {years}: {e}")
        return None

    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2"]]
    bmx = bmx[["SEQN", "BMXBMI", "BMXWAIST"]]

    sbp_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in bpx.columns]
    dbp_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in bpx.columns]
    bpx = bpx.assign(
        systolic_bp_mmhg=bpx[sbp_cols].mean(axis=1) if sbp_cols else pd.NA,
        diastolic_bp_mmhg=bpx[dbp_cols].mean(axis=1) if dbp_cols else pd.NA,
    )[["SEQN", "systolic_bp_mmhg", "diastolic_bp_mmhg"]]

    smq = smq[["SEQN", "SMQ020"]].rename(columns={"SMQ020": "EVER_SMOKED"})

    df = (
        demo.merge(bmx, on="SEQN", how="left")
            .merge(bpx, on="SEQN", how="left")
            .merge(smq, on="SEQN", how="left")
            .merge(mort, on="SEQN", how="inner")  # inner join keeps only records present on the LMF
    )
    df["cycle"] = years
    return df


def build_dataset() -> pd.DataFrame:
    frames = []
    for cycle in CYCLES:
        df = process_cycle(cycle)
        if df is not None:
            frames.append(df)
        time.sleep(1)  # be polite to the CDC server

    full = pd.concat(frames, ignore_index=True)

    # Keep only participants who were eligible for mortality linkage and
    # have a known vital status + follow-up time
    full = full[full["ELIGSTAT"] == 1]
    full = full.dropna(subset=["MORTSTAT", "PERMTH_EXM"])

    full = full.rename(columns={
        "SEQN": "participant_id",
        "RIDAGEYR": "age",
        "RIAGENDR": "sex",
        "RIDRETH1": "race_eth",
        "DMDEDUC2": "education",
        "BMXBMI": "bmi",
        "BMXWAIST": "waist_cm",
        "MORTSTAT": "event",
        "PERMTH_EXM": "months_in_study",
    })

    # Replace numeric codes with human-readable labels. EVER_SMOKED can be
    # missing (kids under 20 were not asked), so it is mapped through the
    # nullable Int64 dtype rather than plain int.
    full["sex"] = full["sex"].astype(int).map(SEX_LABELS)
    full["race_eth"] = full["race_eth"].astype(int).map(RACE_ETH_LABELS)
    full["EVER_SMOKED"] = full["EVER_SMOKED"].astype("Int64").map(SMOKED_LABELS)
    full["event"] = full["event"].astype(int).map(EVENT_LABELS)

    keep_cols = [
        "participant_id", "cycle", "age", "sex", "race_eth", "education",
        "bmi", "waist_cm", "systolic_bp_mmhg", "diastolic_bp_mmhg",
        "EVER_SMOKED", "months_in_study", "event",
    ]
    return full[keep_cols].reset_index(drop=True)


def build_5yr_horizon_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the survival dataset into a fixed 5-year mortality outcome.

    A row's status at the horizon is only known if the participant either
    died within 5 years, or was followed for at least 5 years. Rows
    censored earlier than 5 years (lost to follow-up before the horizon)
    are dropped, since we cannot tell if they died or survived past it.
    """
    known_at_horizon = (
        (df["event"] == "deceased") | (df["months_in_study"] >= HORIZON_MONTHS)
    )
    horizon_df = df[known_at_horizon].copy()

    horizon_df["death_within_5y"] = (
        (horizon_df["event"] == "deceased")
        & (horizon_df["months_in_study"] <= HORIZON_MONTHS)
    ).astype(int)

    # "participant_id" is just an identifier, "event" is the raw vital
    # status that "death_within_5y" was derived from (so keeping it would
    # leak the target), and "cycle" is dropped too.
    return horizon_df.drop(columns=["months_in_study", "participant_id", "event", "cycle"])


if __name__ == "__main__":
    import os

    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

    dataset = build_dataset()
    print("\nFinal shape:", dataset.shape)
    print("\nEvent distribution:")
    print(dataset["event"].value_counts())
    out_path = os.path.join(examples_dir, "nhanes_1999_2018_mortality.csv")
    dataset.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    horizon_dataset = build_5yr_horizon_dataset(dataset)
    print("\n5-year horizon shape:", horizon_dataset.shape)
    print("\n5-year mortality distribution:")
    print(horizon_dataset["death_within_5y"].value_counts())
    horizon_out_path = os.path.join(examples_dir, "nhanes_1999_2018_mortality_5yr_horizon.csv")
    horizon_dataset.to_csv(horizon_out_path, index=False)
    print(f"\nSaved to {horizon_out_path}")
