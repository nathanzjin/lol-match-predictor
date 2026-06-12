"""
Fetch the Oracle's Elixir match-data CSVs into data/raw/.

Oracle's Elixir distributes its yearly CSVs via Google Drive. This script pulls
them with gdown so the pipeline is reproducible from a fresh clone.

Usage:
    python download_data.py                       # uses DEFAULT_FOLDER_URL below
    python download_data.py --folder <drive_url>  # use a different Drive folder

Notes:
    - Heavily-downloaded public Drive files can hit a "quota exceeded" error.
      If that happens, make your own copy of the folder in Google Drive
      ("Make a copy"), share it as "anyone with the link", and pass that URL.
    - Data is intentionally git-ignored (large + reproducible).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import gdown

# A Google Drive folder containing the yearly OE CSVs (2023-2026).
# Replace with your own copy if this one is rate-limited.
DEFAULT_FOLDER_URL = "https://drive.google.com/drive/folders/1IOba4ltJEJ8W7VYntrZEP4K-Tg2xAEEd"

DEST = Path("data/raw")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Oracle's Elixir CSVs into data/raw/")
    parser.add_argument("--folder", default=DEFAULT_FOLDER_URL,
                        help="Google Drive folder URL containing the yearly CSVs")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OE CSVs into {DEST}/ ...")
    gdown.download_folder(url=args.folder, output=str(DEST), quiet=False, use_cookies=False)
    print("Done. Files in data/raw/:")
    for f in sorted(DEST.glob("*.csv")):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
