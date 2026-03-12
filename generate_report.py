#!/usr/bin/env python3
"""
generate_report.py
==================
Converts REPORT.md → REPORT.pdf using pandoc + XeLaTeX.

Usage
-----
  # Generate once
  python generate_report.py

  # Watch mode: auto-regenerate whenever REPORT.md is saved
  python generate_report.py --watch

  # Custom paths
  python generate_report.py --input REPORT.md --output docs/VLA_Report.pdf

Requirements (already available on this machine)
------------------------------------------------
  pandoc   >= 3.9   (brew install pandoc)
  xelatex           (MacTeX / TeX Live)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Pandoc / LaTeX configuration ──────────────────────────────────────────────

# Minimal YAML front matter — no header-includes to avoid LaTeX injection issues.
# All styling is handled via pandoc CLI -V variables below.
YAML_FRONTMATTER = """\
---
title: "VLA Robot Learning — Project Report"
author: "Sara Aly"
date: "{date}"
---

"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def update_last_updated(md_path: Path) -> None:
    """Rewrite the 'Last Updated:' line in REPORT.md with today's date."""
    today = datetime.now().strftime("%Y-%m-%d")
    content = md_path.read_text(encoding="utf-8")
    # Match: **Last Updated:** 2026-02-22  or  Last Updated: 2026-02-22
    new_content = re.sub(
        r"(\*\*Last Updated:\*\*\s*)[\d\-]+",
        rf"\g<1>{today}",
        content,
    )
    if new_content != content:
        md_path.write_text(new_content, encoding="utf-8")


def update_changelog(md_path: Path, entry: str) -> None:
    """
    Append a dated entry to the Change Log table in REPORT.md.
    Call this from other scripts after making significant changes.

    Example:
        update_changelog(Path("REPORT.md"), "Added PPO trainer to training/ppo_trainer.py")
    """
    today = datetime.now().strftime("%Y-%m-%d")
    content = md_path.read_text(encoding="utf-8")
    new_row = f"| {today} | {entry} |"

    # Find the last row of the changelog table and insert after it
    # The table ends before the next blank line after the last | line
    lines = content.splitlines()
    last_table_line = -1
    in_changelog = False
    for i, line in enumerate(lines):
        if "## 11. Change Log" in line or "## Change Log" in line:
            in_changelog = True
        if in_changelog and line.startswith("|"):
            last_table_line = i

    if last_table_line >= 0:
        lines.insert(last_table_line + 1, new_row)
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[report] Changelog updated: {new_row}")
    else:
        print("[report] Warning: Could not find Change Log table to update.")


def build_pdf(md_path: Path, pdf_path: Path) -> bool:
    """
    Prepend minimal YAML front matter to a temp file, then run pandoc
    with styling passed as -V variables. Returns True on success.
    """
    date_str = datetime.now().strftime("%B %d, %Y")

    # Read original markdown, strip any existing YAML front matter
    raw = md_path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            raw = raw[end + 4:].lstrip("\n")

    # Write temp file with clean YAML header
    tmp_path = md_path.parent / f"_tmp_report_{os.getpid()}.md"
    tmp_path.write_text(
        YAML_FRONTMATTER.format(date=date_str) + raw,
        encoding="utf-8",
    )

    cmd = [
        "pandoc",
        str(tmp_path),
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--standalone",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "colorlinks=true",
        "-V", "linkcolor=NavyBlue",
        "-V", "urlcolor=NavyBlue",
        "-V", "citecolor=NavyBlue",
        "-V", "mainfont=Helvetica Neue",
        "-V", "monofont=Menlo",
        "-V", "linestretch=1.25",
        "-V", "tables-rules=true",
        "-o", str(pdf_path),
    ]

    print(f"[report] Generating {pdf_path.name} …")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=md_path.parent,
        )
        if result.returncode == 0:
            size_kb = pdf_path.stat().st_size / 1024
            print(f"[report] ✓ {pdf_path.name} ({size_kb:.1f} KB)")
            return True
        else:
            print(f"[report] ✗ pandoc failed:\n{result.stderr}")
            return False
    except FileNotFoundError:
        print("[report] ✗ pandoc not found. Install with: brew install pandoc")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def generate(md_path: Path, pdf_path: Path) -> bool:
    """Update timestamps, then build PDF."""
    update_last_updated(md_path)
    return build_pdf(md_path, pdf_path)


def watch(md_path: Path, pdf_path: Path, interval: float = 2.0) -> None:
    """
    Poll REPORT.md every `interval` seconds.
    Re-generate the PDF whenever the file modification time changes.
    Press Ctrl+C to stop.
    """
    print(f"[report] Watching {md_path.name} for changes (Ctrl+C to stop) …")
    last_mtime = md_path.stat().st_mtime
    generate(md_path, pdf_path)  # build immediately on start

    try:
        while True:
            time.sleep(interval)
            mtime = md_path.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                print(f"\n[report] Change detected — rebuilding …")
                generate(md_path, pdf_path)
    except KeyboardInterrupt:
        print("\n[report] Watch stopped.")


# ── LaTeX compiler ────────────────────────────────────────────────────────────

def build_latex(tex_path: Path, pdf_path: Path) -> bool:
    """
    Compile REPORT.tex → PDF using xelatex (run twice for TOC + references).
    BibTeX is run between the two xelatex passes if references.bib exists.
    Output PDF is copied to pdf_path.
    """
    here = tex_path.parent
    base = tex_path.stem          # "REPORT"

    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True, cwd=here)

    print(f"[report] Compiling {tex_path.name} (pass 1) …")
    r1 = run(["xelatex", "-interaction=nonstopmode", tex_path.name])
    if r1.returncode != 0:
        # Print last 20 lines of log for quick diagnosis
        log_file = here / f"{base}.log"
        if log_file.exists():
            lines = log_file.read_text(errors="replace").splitlines()
            print("\n".join(lines[-20:]))
        else:
            print(r1.stdout[-2000:])
        print("[report] ✗ xelatex pass 1 failed.")
        return False

    # BibTeX pass
    bib_path = here / "references.bib"
    if bib_path.exists():
        print("[report] Running bibtex …")
        run(["bibtex", base])

    print(f"[report] Compiling {tex_path.name} (pass 2 — TOC + refs) …")
    r2 = run(["xelatex", "-interaction=nonstopmode", tex_path.name])
    if r2.returncode != 0:
        print("[report] ✗ xelatex pass 2 failed.")
        return False

    # Move output PDF to requested path
    built_pdf = here / f"{base}.pdf"
    if built_pdf.exists():
        if pdf_path != built_pdf:
            import shutil
            shutil.copy2(built_pdf, pdf_path)
        size_kb = pdf_path.stat().st_size / 1024
        print(f"[report] ✓ {pdf_path.name} ({size_kb:.1f} KB)")
        return True

    print("[report] ✗ No PDF produced.")
    return False


def watch_latex(tex_path: Path, pdf_path: Path, interval: float = 2.0) -> None:
    """Watch REPORT.tex (and references.bib) for changes; recompile on save."""
    watch_files = [tex_path, tex_path.parent / "references.bib"]
    mtimes = {f: f.stat().st_mtime for f in watch_files if f.exists()}

    print(f"[report] Watching {tex_path.name} for changes (Ctrl+C to stop) …")
    build_latex(tex_path, pdf_path)
    try:
        while True:
            time.sleep(interval)
            changed = False
            for f in watch_files:
                if f.exists():
                    mt = f.stat().st_mtime
                    if mt != mtimes.get(f):
                        mtimes[f] = mt
                        changed = True
            if changed:
                print(f"\n[report] Change detected — recompiling …")
                build_latex(tex_path, pdf_path)
    except KeyboardInterrupt:
        print("\n[report] Watch stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate REPORT.pdf from REPORT.tex (--tex) or REPORT.md (default)"
    )
    parser.add_argument(
        "--tex", action="store_true",
        help="Compile REPORT.tex with xelatex instead of converting REPORT.md",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Override input file path",
    )
    parser.add_argument(
        "--output", "-o",
        default="REPORT.pdf",
        help="Output PDF path (default: REPORT.pdf)",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch for changes and auto-regenerate",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Poll interval in seconds for --watch mode (default: 2)",
    )
    parser.add_argument(
        "--log", "-l",
        metavar="ENTRY",
        help="Append a changelog entry to REPORT.md (markdown mode only)",
    )
    args = parser.parse_args()

    here     = Path(__file__).parent
    pdf_path = (here / args.output).resolve()

    # ── LaTeX mode ────────────────────────────────────────────────────────────
    if args.tex:
        tex_path = (here / (args.input or "REPORT.tex")).resolve()
        if not tex_path.exists():
            print(f"[report] Error: {tex_path} not found.")
            sys.exit(1)
        if args.watch:
            watch_latex(tex_path, pdf_path, interval=args.interval)
        else:
            ok = build_latex(tex_path, pdf_path)
            sys.exit(0 if ok else 1)
        return

    # ── Markdown mode (default) ───────────────────────────────────────────────
    md_path = (here / (args.input or "REPORT.md")).resolve()

    if not md_path.exists():
        print(f"[report] Error: {md_path} not found.")
        sys.exit(1)

    if args.log:
        update_changelog(md_path, args.log)

    if args.watch:
        watch(md_path, pdf_path, interval=args.interval)
    else:
        ok = generate(md_path, pdf_path)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
