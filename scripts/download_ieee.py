"""Download IEEE-CIS Fraud Detection, or the ULB fallback.

    python scripts/download_ieee.py            # IEEE-CIS (~590K labelled transactions)
    python scripts/download_ieee.py --ulb      # ULB credit-card fraud (smaller fallback)

WHY THIS SCRIPT EXISTS
----------------------
An optional validation is only honest if it can actually run. Two halves make it so:

  1. data/real_calibration.py returns an explicit `not_validated` state when the data is
     absent, never a pass.
  2. This script makes the download step documented and executable, so the check can
     genuinely run rather than being permanently skipped.

SETUP
-----
  1. Free Kaggle account: https://www.kaggle.com
  2. Settings -> API -> "Create New Token" -> kaggle.json
  3. Place at ~/.kaggle/kaggle.json (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)
  4. pip install kaggle
  5. IEEE-CIS is a COMPETITION dataset: you must accept its rules once, while signed in,
     at https://www.kaggle.com/c/ieee-fraud-detection/rules
     Skipping step 5 causes a 403 even with a valid token. The ULB fallback
     (`--ulb`) is an ordinary dataset and needs no rule acceptance.
  6. python scripts/download_ieee.py

The pipeline runs fine WITHOUT this — the model card reports `not_validated`, which is
the honest state, not a pass.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile

RAW_DIR = "data/raw"
IEEE_COMP = "ieee-fraud-detection"
ULB_DATASET = "mlg-ulb/creditcardfraud"


def _preflight() -> int | None:
    if shutil.which("kaggle") is None:
        print("ERROR: the `kaggle` CLI is not installed.\n       pip install kaggle",
              file=sys.stderr)
        return 1
    cred = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(cred):
        print(f"ERROR: no Kaggle API token at {cred}\n"
              "       Kaggle -> Settings -> API -> Create New Token.", file=sys.stderr)
        return 1
    return None


def download_ieee() -> int:
    target = os.path.join(RAW_DIR, "ieee_fraud.csv")
    if os.path.exists(target):
        print(f"{target} already present — nothing to do.")
        return 0

    print(f"Downloading competition data: {IEEE_COMP} ...")
    r = subprocess.run(
        ["kaggle", "competitions", "download", "-c", IEEE_COMP, "-p", RAW_DIR],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        msg = (r.stderr or "") + (r.stdout or "")
        print(f"ERROR: download failed:\n{msg}", file=sys.stderr)
        if "403" in msg or "Forbidden" in msg:
            print("\nA 403 almost always means the competition rules have not been\n"
                  "accepted. Sign in and accept them at:\n"
                  f"  https://www.kaggle.com/c/{IEEE_COMP}/rules\n"
                  "Or use the smaller fallback:  python scripts/download_ieee.py --ulb",
                  file=sys.stderr)
        return r.returncode

    archive = os.path.join(RAW_DIR, f"{IEEE_COMP}.zip")
    if os.path.exists(archive):
        print("Extracting ...")
        with zipfile.ZipFile(archive) as z:
            z.extractall(RAW_DIR)
        os.remove(archive)

    # The validator reads a single file; train_transaction.csv carries the label and the
    # feature families we check.
    src = os.path.join(RAW_DIR, "train_transaction.csv")
    if os.path.exists(src):
        os.replace(src, target)
        print(f"Renamed train_transaction.csv -> {target}")
    else:
        print("WARNING: train_transaction.csv not found after extraction; "
              f"rename the labelled file to {target} manually.", file=sys.stderr)
        return 1

    print("Done. Re-run `python run.py` — IEEE validation will now report 'validated'.")
    return 0


def download_ulb() -> int:
    target = os.path.join(RAW_DIR, "creditcard.csv")
    if os.path.exists(target):
        print(f"{target} already present — nothing to do.")
        return 0

    print(f"Downloading {ULB_DATASET} ...")
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", ULB_DATASET, "-p", RAW_DIR],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"ERROR: download failed:\n{r.stderr}", file=sys.stderr)
        return r.returncode

    archive = os.path.join(RAW_DIR, "creditcardfraud.zip")
    if os.path.exists(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(RAW_DIR)
        os.remove(archive)

    print("Done. Note: ULB is fully anonymised PCA features, so it can only validate the\n"
          "transaction-value family. Account-tenure, velocity and identity-collision\n"
          "families need IEEE-CIS. The model card records which one actually ran.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ulb", action="store_true",
                    help="download the smaller ULB credit-card dataset instead")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    bad = _preflight()
    if bad is not None:
        return bad
    return download_ulb() if args.ulb else download_ieee()


if __name__ == "__main__":
    raise SystemExit(main())
