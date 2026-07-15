"""
Build a large ICU sepsis-prediction dataset from the public PhysioNet /
Computing in Cardiology Challenge 2019 (early prediction of sepsis):
40,336 ICU patients, one pipe-delimited file per patient with hourly
vitals and labs, and an hourly sepsis flag.

For each patient, we summarize their first 24 hours in the ICU by the
mean of each vital/lab, and label them "sepsis" if they are ever
flagged septic *later* in their stay. Patients whose sepsis onset falls
inside that first 24h window are dropped: for them, the "first 24h"
features would already contain data from during the event itself,
rather than strictly preceding it, which would make the prediction task
trivial and not representative of an early-warning use case.

Source (public, no credentialing required):
    https://physionet.org/content/challenge-2019/1.0.0/

This downloads all 40,336 individual patient files (a few hundred bytes
to a few KB each) with a pool of worker threads, since downloading them
one at a time would take a very long time.

Output columns:
    age                   - age in years
    sex                   - "M" or "F"
    hours_before_icu      - hours between hospital and ICU admission
                            (negative = ICU admission preceded the
                            recorded hospital admission time)
    heart_rate_bpm        - mean over the first 24h in the ICU
    o2_sat_pct            - mean over the first 24h
    temp_celsius          - mean over the first 24h
    systolic_bp_mmhg      - mean over the first 24h
    diastolic_bp_mmhg     - mean over the first 24h
    mean_arterial_bp_mmhg - mean over the first 24h
    resp_rate             - mean over the first 24h
    glucose_mgdl          - mean over the first 24h
    potassium_mEqL        - mean over the first 24h
    hematocrit_pct        - mean over the first 24h
    wbc_count             - mean over the first 24h
    creatinine_mgdl       - mean over the first 24h
    bun_mgdl               - mean over the first 24h
    platelets_count        - mean over the first 24h
    lactate_mmolL           - mean over the first 24h
    hemoglobin_gdl           - mean over the first 24h
    sepsis                   - 1 if the patient is later flagged septic,
                               0 otherwise

This saves two CSVs:
    - physionet_sepsis_full.csv, with every column above, saved next to
      this script (the project root). This is the reference dataset,
      kept out of examples/ so it does not get shipped to JupyterLite.
    - examples/physionet_sepsis.csv, a lighter version with only the
      columns the example scripts actually use (age, sex,
      hours_before_icu, diastolic_bp_mmhg, sepsis). This is the one
      that gets shipped alongside the notebooks.

Requires: pandas, requests
    pip install pandas requests

Run from the health_ml_tutorial directory:
    python download_physionet_sepsis.py
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training"
TRAINING_SETS = ["training_setA", "training_setB"]
N_WORKERS = 25

# The full download is ~40,000 small requests; on a flaky connection some
# will time out transiently. Retry a few times with backoff before giving
# up on a given file, and only skip that one file rather than letting one
# persistent failure crash the whole download.
SESSION = requests.Session()
RETRY = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
SESSION.mount("https://", HTTPAdapter(max_retries=RETRY))

RENAME = {
    "Age": "age",
    "HospAdmTime": "hours_before_icu",
    "HR": "heart_rate_bpm",
    "O2Sat": "o2_sat_pct",
    "Temp": "temp_celsius",
    "SBP": "systolic_bp_mmhg",
    "DBP": "diastolic_bp_mmhg",
    "MAP": "mean_arterial_bp_mmhg",
    "Resp": "resp_rate",
    "Glucose": "glucose_mgdl",
    "Potassium": "potassium_mEqL",
    "Hct": "hematocrit_pct",
    "WBC": "wbc_count",
    "Creatinine": "creatinine_mgdl",
    "BUN": "bun_mgdl",
    "Platelets": "platelets_count",
    "Lactate": "lactate_mmolL",
    "Hgb": "hemoglobin_gdl",
}

MEAN_COLUMNS = [
    "Age", "HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp",
    "Glucose", "Potassium", "Hct", "WBC", "Creatinine", "BUN",
    "Platelets", "Lactate", "Hgb",
]

# The columns actually used by the example scripts in examples/
LIGHT_COLUMNS = ["age", "sex", "hours_before_icu", "diastolic_bp_mmhg", "sepsis"]


def list_patient_files(training_set):
    """List the .psv patient file names in one training set, by reading
    the directory listing page."""
    r = SESSION.get(f"{BASE_URL}/{training_set}/", timeout=60)
    r.raise_for_status()
    return re.findall(r'href="(p\d+\.psv)"', r.text)


def fetch_patient_row(training_set, filename):
    """Download one patient file and summarize it into a single row, or
    return None if the download ultimately fails or the patient's sepsis
    onset falls in the first 24h."""
    url = f"{BASE_URL}/{training_set}/{filename}"
    try:
        r = SESSION.get(url, timeout=60)
        r.raise_for_status()
        patient = pd.read_csv(StringIO(r.text), sep="|")
    except (requests.RequestException, pd.errors.ParserError) as e:
        print(f"  Skipped {filename}: {e}")
        return None

    is_septic = patient["SepsisLabel"].max() == 1
    if is_septic:
        onset_hour = patient.loc[patient["SepsisLabel"] == 1, "ICULOS"].min()
        if onset_hour <= 24:
            return None

    first_24h = patient[patient["ICULOS"] <= 24]
    row = first_24h[MEAN_COLUMNS].mean().to_dict()
    row["Gender"] = patient["Gender"].iloc[0]
    row["HospAdmTime"] = patient["HospAdmTime"].iloc[0]
    row["sepsis"] = int(is_septic)
    return row


if __name__ == "__main__":
    rows = []
    for training_set in TRAINING_SETS:
        filenames = list_patient_files(training_set)
        print(f"{training_set}: {len(filenames)} patient files, downloading...")

        def fetch(filename):
            return fetch_patient_row(training_set, filename)

        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            for row in pool.map(fetch, filenames):
                if row is not None:
                    rows.append(row)

        print(f"{training_set}: done, {len(rows)} patients kept so far")

    df = pd.DataFrame(rows)
    df["sex"] = df["Gender"].map({0: "F", 1: "M"})
    df = df.drop(columns=["Gender"])
    df = df.rename(columns=RENAME)

    print("\nFinal shape:", df.shape)
    print("\nSepsis distribution:")
    print(df["sepsis"].value_counts())

    project_root = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(project_root, "examples")

    full_out_path = os.path.join(project_root, "physionet_sepsis_full.csv")
    df.to_csv(full_out_path, index=False)
    print(f"\nSaved full dataset to {full_out_path}")

    light_out_path = os.path.join(examples_dir, "physionet_sepsis.csv")
    df[LIGHT_COLUMNS].to_csv(light_out_path, index=False)
    print(f"Saved light dataset (used by the examples) to {light_out_path}")
