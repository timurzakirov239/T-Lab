"""Protect completed experiment evidence from accidental reuse of its directory."""
from __future__ import annotations

import json
from pathlib import Path


def ensure_run_writable(output: Path) -> None:
    """Refuse to overwrite any completed or explicitly frozen experiment."""
    output = Path(output)
    if (output / "FROZEN.json").exists():
        raise FileExistsError(f"Frozen experiment: {output}. Use a new run ID and output directory.")
    status = output / "run_status.json"
    if status.exists():
        try:
            current = json.loads(status.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise FileExistsError(f"Unreadable experiment status: {status}. Use a new output directory.") from None
        if not isinstance(current, dict) or current.get("status") == "complete":
            raise FileExistsError(f"Completed experiment: {output}. Use a new run ID and output directory.")
