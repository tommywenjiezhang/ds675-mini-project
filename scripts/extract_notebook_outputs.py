#!/usr/bin/env python3
"""Extract output-producing cells from a Jupyter notebook into JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nbformat


def _output_to_text(output: dict[str, Any]) -> str:
    """Return a compact text representation of a notebook output."""
    output_type = output.get("output_type", "")

    if output_type == "stream":
        return str(output.get("text", "")).strip()

    if output_type in {"execute_result", "display_data"}:
        data = output.get("data", {})
        if "text/plain" in data:
            text = data["text/plain"]
            if isinstance(text, list):
                return "".join(text).strip()
            return str(text).strip()
        return f"[non-text output keys: {', '.join(sorted(data.keys()))}]"

    if output_type == "error":
        traceback = output.get("traceback", [])
        if traceback:
            return "\n".join(traceback).strip()
        return f"{output.get('ename', '')}: {output.get('evalue', '')}".strip()

    return ""


def extract_outputs(notebook_path: Path) -> list[dict[str, Any]]:
    nb = nbformat.read(notebook_path, as_version=4)
    extracted: list[dict[str, Any]] = []

    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        extracted.append(
            {
                "cell_index": idx,
                "execution_count": cell.get("execution_count"),
                "source": cell.get("source", ""),
                "num_outputs": len(outputs),
                "outputs": [
                    {
                        "output_type": out.get("output_type", ""),
                        "summary": _output_to_text(out),
                    }
                    for out in outputs
                ],
            }
        )

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all output-producing cells from a notebook into JSON."
    )
    parser.add_argument(
        "--notebook",
        default="milestone4_report.ipynb",
        help="Path to input notebook (default: milestone4_report.ipynb)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/milestone4_output_cells.json",
        help="Path to output JSON file (default: artifacts/milestone4_output_cells.json)",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    output_path = Path(args.output)

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    data = {
        "notebook": str(notebook_path),
        "output_cell_count": 0,
        "output_cells": [],
    }

    output_cells = extract_outputs(notebook_path)
    data["output_cell_count"] = len(output_cells)
    data["output_cells"] = output_cells

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"Extracted {len(output_cells)} output-producing cells")
    print(f"Saved JSON to: {output_path}")


if __name__ == "__main__":
    main()
