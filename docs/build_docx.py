"""Build a Word (.docx) version of docs/QAQC.md.

Steps
-----
1. Extract every ```mermaid ...``` fenced block from QAQC.md.
2. Render each one to a PNG via the Kroki service (https://kroki.io).
3. Write a transformed copy of the markdown where each diagram is replaced
   by a numbered figure image reference.
4. Convert the transformed markdown to QAQC.docx with Pandoc, turning the
   ``$$...$$`` LaTeX block(s) into native Word equations.

Run:  python docs/build_docx.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pypandoc
import requests

DOCS = Path(__file__).resolve().parent
SRC = DOCS / "QAQC.md"
ASSETS = DOCS / "_assets"
TRANSFORMED = DOCS / "QAQC.pandoc.md"
OUT = DOCS / "QAQC.docx"

KROKI_URL = "https://kroki.io/mermaid/png"

MERMAID_BLOCK = re.compile(r"```mermaid[ \t]*\n(.*?)\n```", re.DOTALL)


def render_mermaid(code: str, index: int) -> Path:
    """Render one Mermaid diagram to a PNG using Kroki, return the file path."""
    resp = requests.post(
        KROKI_URL,
        data=code.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        timeout=60,
    )
    resp.raise_for_status()
    out = ASSETS / f"qaqc_figure_{index:02d}.png"
    out.write_bytes(resp.content)
    return out


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    text = SRC.read_text(encoding="utf-8")

    counter = {"n": 0}

    def replace(match: re.Match[str]) -> str:
        counter["n"] += 1
        n = counter["n"]
        code = match.group(1)
        print(f"  rendering figure {n} ...", flush=True)
        png = render_mermaid(code, n)
        # Relative, space-free path resolved by Pandoc via --resource-path=docs.
        uri = png.relative_to(DOCS).as_posix()
        return f"![Figure {n}]({uri})"

    transformed = MERMAID_BLOCK.sub(replace, text)
    TRANSFORMED.write_text(transformed, encoding="utf-8")
    print(f"Rendered {counter['n']} Mermaid figures to {ASSETS}")

    pypandoc.convert_file(
        str(TRANSFORMED),
        to="docx",
        format="gfm+tex_math_dollars+implicit_figures",
        outputfile=str(OUT),
        extra_args=[f"--resource-path={DOCS.as_posix()}"],
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
