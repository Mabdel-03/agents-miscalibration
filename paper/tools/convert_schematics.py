#!/usr/bin/env python
"""Rasterise the exported Mermaid schematics in docs/figures/exported/ for inclusion.

No usable SVG-to-PDF converter exists on this machine: the system rsvg-convert is
librsvg 2.42.7, which cannot parse the Mermaid stylesheet and silently emits solid
dark rectangles with no text.  Headless Firefox renders them correctly.

Firefox scales its screenshot to the window width, and these SVGs declare
``width="100%"`` against a fixed viewBox, so screenshotting the SVG directly yields
no resolution gain.  A throwaway wrapper page pins the image to ``scale x viewBox``
width, which does.

Usage:  convert_schematics.py --out figures/schematics --scale 3 --workdir _build
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperdata as D  # noqa: E402

SRC = D.REPO / "docs" / "figures" / "exported"
MIN_BYTES = 60_000


def viewbox(svg_text: str) -> tuple[float, float]:
    m = re.search(r'viewBox="([-\d.eE\s]+)"', svg_text[:4000])
    if not m:
        raise SystemExit("no viewBox found; cannot size the wrapper page")
    parts = [float(x) for x in m.group(1).split()]
    return parts[2], parts[3]


def uniform(png: Path) -> bool:
    """True when the rendered image has almost no colour variation, which is the
    signature of the broken librsvg path."""
    try:
        from PIL import Image
    except ImportError:
        return False
    with Image.open(png) as im:
        small = im.convert("RGB").resize((64, 64))
        cols = small.getcolors(64 * 64) or []
    if not cols:
        return False
    top = max(cols)[0]
    return top / (64 * 64) > 0.995


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/schematics")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--workdir", default="_build")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.workdir)
    stage = work / "svgstage"
    ffhome = work / "ffhome"
    ffprofile = work / "ffprofile"
    for p in (stage, ffhome, ffprofile):
        p.mkdir(parents=True, exist_ok=True)

    firefox = shutil.which("firefox") or "/usr/bin/firefox"
    if not Path(firefox).exists():
        raise SystemExit("firefox not found; cannot rasterise the schematics")

    svgs = sorted(SRC.glob("*.svg"))
    if not svgs:
        raise SystemExit(f"no SVGs under {SRC}")

    env = dict(os.environ)
    env.update({"HOME": str(ffhome.resolve()), "MOZ_HEADLESS": "1"})

    for svg in svgs:
        text = svg.read_text()
        w, h = viewbox(text)
        shutil.copy(svg, stage / svg.name)
        html = stage / f"{svg.stem}.html"
        html.write_text(
            '<html><body style="margin:0;background:#ffffff">'
            f'<img src="{svg.name}" style="width:{w * args.scale:.0f}px;display:block">'
            "</body></html>"
        )
        target = out / f"{svg.stem}.png"
        subprocess.run(
            [
                firefox, "--headless", "--profile", str(ffprofile.resolve()),
                "--window-size", str(int(w * args.scale)),
                "--screenshot", str(target.resolve()),
                html.resolve().as_uri(),
            ],
            env=env, check=True, timeout=180,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not target.exists() or target.stat().st_size < MIN_BYTES:
            raise SystemExit(f"{target} is missing or implausibly small; rasterisation failed")
        if uniform(target):
            raise SystemExit(f"{target} is a flat image; the renderer produced no content")
        print(f"  {target.name}  {target.stat().st_size / 1000:.0f} KB")

    print(f"rasterised {len(svgs)} schematics at {args.scale}x into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
