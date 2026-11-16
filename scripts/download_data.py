"""Download the raw experiment datasets into data/raw/.

Only Hillstrom so far. Files are skipped when already present.
"""

from pathlib import Path
from urllib.request import Request, urlopen

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    # Kevin Hillstrom's MineThatData E-Mail Analytics Challenge (2008), ~4 MB.
    "hillstrom.csv": (
        "http://www.minethatdata.com/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    ),
}


def download(filename: str, url: str) -> Path:
    target = RAW_DIR / filename
    if target.exists():
        print(f"{filename}: already present, skipping")
        return target
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "python-urllib"})
    with urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())
    print(f"{filename}: downloaded {target.stat().st_size:,} bytes")
    return target


if __name__ == "__main__":
    for name, source in DATASETS.items():
        download(name, source)
