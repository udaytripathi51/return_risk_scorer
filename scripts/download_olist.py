"""Download the Olist Brazilian E-Commerce dataset.

    python scripts/download_olist.py

Used ONLY to calibrate order/category distribution *shape*. Olist is Brazilian, so it
tells us nothing about Indian COD/RTO behaviour or about fraud — see
data/real_calibration.py for how narrowly it is used.

SETUP (documented rather than silently skipped):
  1. Create a free Kaggle account at https://www.kaggle.com
  2. Account -> Settings -> API -> "Create New Token". This downloads kaggle.json.
  3. Place it at ~/.kaggle/kaggle.json  (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)
     and restrict permissions:  chmod 600 ~/.kaggle/kaggle.json
  4. pip install kaggle
  5. python scripts/download_olist.py

The pipeline runs fine WITHOUT this. `load_olist_calibration()` reports
`status='not_calibrated'` and falls back to the documented defaults in config.py. The
point is that the fallback is visible, never disguised as success.

Sourced via the Kaggle API deliberately, not an unverified raw-GitHub mirror.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile

DATASET = "olistbr/brazilian-ecommerce"
RAW_DIR = "data/raw"
WANTED = [
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_orders_dataset.csv",
]


def main() -> int:
    os.makedirs(RAW_DIR, exist_ok=True)

    if all(os.path.exists(os.path.join(RAW_DIR, f)) for f in WANTED[:2]):
        print(f"Already present in {RAW_DIR}/ — nothing to do.")
        return 0

    if shutil.which("kaggle") is None:
        print("ERROR: the `kaggle` CLI is not installed.\n"
              "       pip install kaggle\n"
              "       then follow the SETUP steps in this file's docstring.",
              file=sys.stderr)
        return 1

    cred = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(cred):
        print(f"ERROR: no Kaggle API token at {cred}\n"
              "       Kaggle -> Settings -> API -> Create New Token, then place the\n"
              "       downloaded kaggle.json there.", file=sys.stderr)
        return 1

    print(f"Downloading {DATASET} into {RAW_DIR}/ ...")
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", RAW_DIR],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"ERROR: kaggle download failed:\n{r.stderr}", file=sys.stderr)
        return r.returncode

    archive = os.path.join(RAW_DIR, "brazilian-ecommerce.zip")
    if os.path.exists(archive):
        print("Extracting ...")
        with zipfile.ZipFile(archive) as z:
            z.extractall(RAW_DIR)
        os.remove(archive)

    found = [f for f in WANTED if os.path.exists(os.path.join(RAW_DIR, f))]
    print(f"Done. Files available: {found}")
    print("Re-run `python run.py` — Olist calibration will now report 'calibrated'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
