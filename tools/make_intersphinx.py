#!/usr/bin/env python3
"""
Fetch intersphinx inventories into docs/_intersphinx (by default).

Usage:
  python fetch_intersphinx.py
  python fetch_intersphinx.py --target /path/to/_intersphinx
  python fetch_intersphinx.py --add pandas=https://pandas.pydata.org/docs/objects.inv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

DEFAULT_SOURCES = {
    "python":      "https://docs.python.org/3/objects.inv",
    "numpy":       "https://numpy.org/doc/stable/objects.inv",
    "matplotlib":  "https://matplotlib.org/stable/objects.inv",
    "pytest":      "https://docs.pytest.org/en/stable/objects.inv",
    "psi-data-utils": "https://predsci.com/doc/psi-data-utils/objects.inv",
    "scipy":       "https://docs.scipy.org/doc/scipy/objects.inv",
}

def fetch(url: str, dest: Path, timeout: float = 30.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "fetch-intersphinx/1.0"})
    try:
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
    except HTTPError as e:
        print(f"HTTP {e.code} for {url}", file=sys.stderr)
        raise
    except URLError as e:
        print(f"URL error for {url}: {e}", file=sys.stderr)
        raise
    dest.write_bytes(data)
    print(f"✓ {url}  ->  {dest}")

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    default_target = parent_dir / "docs" / "_intersphinx"

    ap = argparse.ArgumentParser(description="Download intersphinx inventories.")
    ap.add_argument("--target", type=Path, default=default_target,
                    help=f"Output directory (default: {default_target})")
    ap.add_argument(
        "--add", action="append", default=[],
        help="Extra inventory as NAME=URL (can be repeated).",
    )
    return ap.parse_args()

def main() -> int:
    args = parse_args()

    sources = dict(DEFAULT_SOURCES)
    # Handle --add NAME=URL
    for spec in args.add:
        if "=" not in spec:
            print(f"--add expects NAME=URL, got: {spec}", file=sys.stderr)
            return 2
        name, url = spec.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name or not url:
            print(f"--add expects NAME=URL, got: {spec}", file=sys.stderr)
            return 2
        sources[name] = url

    target_dir = args.target
    target_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for name, url in sources.items():
        out = target_dir / f"{name}-objects.inv"
        try:
            fetch(url, out)
        except Exception:
            failures.append((name, url))

    if failures:
        print("\nSome downloads failed:", file=sys.stderr)
        for n, u in failures:
            print(f"  - {n}: {u}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
