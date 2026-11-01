"""Download a Hugging Face model to a local folder with resumable HTTP requests.

huggingface_hub's own downloader stalled on this network, so this fetches
every repo file with plain `requests`, resuming partial files with HTTP Range
requests. Progress is printed every ~50 MB.

Usage (from src/): python -m collection.download_model tiiuae/Falcon-H1-1.5B-Instruct
Files go to ~/models/<repo name>/ (outside the OneDrive-synced project).
"""
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url, list_repo_files

MODELS_DIR = Path.home() / "models"


def fetch(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # already complete
            part.rename(out)
            return
        r.raise_for_status()
        total = have + int(r.headers.get("content-length", 0))
        mode = "ab" if r.status_code == 206 else "wb"
        start, last = time.time(), have
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                have += len(chunk)
                if have - last >= 50 << 20:
                    speed = (have - last) / 1e6 / (time.time() - start)
                    print(f"  {out.name}: {have / 1e6:,.0f}/{total / 1e6:,.0f} MB ({speed:.1f} MB/s)", flush=True)
                    start, last = time.time(), have
    part.rename(out)


def main(repo: str) -> None:
    target = MODELS_DIR / repo.split("/")[-1]
    for name in list_repo_files(repo):
        out = target / name
        if out.exists():
            continue
        for attempt in range(20):  # resume after network drops
            try:
                fetch(hf_hub_url(repo, name), out)
                break
            except (requests.RequestException, OSError) as e:
                print(f"  retry {attempt + 1} for {name}: {type(e).__name__}", flush=True)
                time.sleep(10)
        print(f"done {name}", flush=True)
    print(f"Model saved to {target}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiiuae/Falcon-H1-1.5B-Instruct")
