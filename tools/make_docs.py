#!/usr/bin/env python3
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), f"(cwd={cwd})" if cwd else "")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def main() -> int:
    # SCRIPT_DIR = directory containing this script
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    docs_dir = parent_dir / "docs"
    build_dir = docs_dir / "_build"
    autodoc_dir = docs_dir / "source" / "autodoc"
    gallery_dir = docs_dir / "source" / "gallery"

    ap = argparse.ArgumentParser(description="Build Sphinx docs with optional cleanup/fetch steps.")
    ap.add_argument("-c", "--clean", action="store_true", help="Remove previous build/autodoc/gallery")
    ap.add_argument("--use-make", action="store_true", help="Use `make -C docs html` instead of sphinx-build")
    ap.add_argument("--sphinx-args", nargs=argparse.REMAINDER,
                    help="Extra args passed to sphinx-build after `html` (use after `--`)")
    args = ap.parse_args()

    if args.clean:
        for p in (build_dir, autodoc_dir, gallery_dir):
            if p.exists():
                print(f"Removing {p}")
                shutil.rmtree(p, ignore_errors=True)

    # Build docs
    if args.use_make:
        # Uses your Makefile (POSIX). On Windows, prefer sphinx-build below.
        run(["make", "-C", str(docs_dir), "html"])
    else:
        # Direct call to sphinx-build; more portable and explicit
        # Equivalent to: sphinx-build -W --keep-going -b html docs/source docs/_build/html
        out_html = build_dir / "html"
        out_html.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "sphinx",
            "--keep-going",
            "-b", "html",
            str(docs_dir / "source"),
            str(out_html),
        ]
        if args.sphinx_args:
            cmd.extend(args.sphinx_args)
        run(cmd)

    print("Docs build completed.")
    return 0

if __name__ == "__main__":
    SystemExit(main())