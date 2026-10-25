"""Verify portable P003 evidence and any locally retained large outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
SNAPSHOT = TASK / "report/p003_snapshot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((SNAPSHOT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["tracked_evidence"].items():
        path = SNAPSHOT / name
        assert path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"], name
    gate = json.loads((SNAPSHOT / "equivalence.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass" and gate["cells_compared"] == 17_001_600
    assert gate["numpy_array_equal_all_cells"] and gate["query_candidate_order_identical"] and gate["feature_schema_identical"]
    assert len(set(gate["matrix_values_sha256"])) == len(set(gate["matrix_file_sha256"])) == 1
    baseline = json.loads((SNAPSHOT / "p003_equivalence_baseline/metrics.json").read_text(encoding="utf-8"))
    optimized = json.loads((SNAPSHOT / "p003_equivalence_optimized/metrics.json").read_text(encoding="utf-8"))
    assert baseline["input_sha256"] == optimized["input_sha256"]
    assert baseline["config_sha256"] == optimized["config_sha256"]
    for status in (baseline, optimized):
        assert status["status"] == "complete" and not status["profiled"]
        assert not status["model_training"] and not status["final_test_evaluated"]
        assert status["build"]["shape"] == [184800, 92]
    scale = json.loads((SNAPSHOT / "p003_scale256/metrics.json").read_text(encoding="utf-8"))
    assert scale["status"] in ("complete", "failed")
    assert not scale["model_training"] and not scale["final_test_evaluated"]
    assert scale["equivalence_gate"]["status"] == "pass"
    if scale["status"] == "complete":
        assert scale["build"]["shape"] == [scale["candidates"]["candidate_rows"], 92]
    else:
        assert scale.get("error")
    local_checks = 0
    for relative, expected in manifest["local_outputs"].items():
        path = TASK / relative
        if path.exists():
            assert sha256(path) == expected["sha256"], relative
            local_checks += 1
    left = TASK / "artifacts/p003_equivalence_baseline/features.npy"
    right = TASK / "artifacts/p003_equivalence_optimized/features.npy"
    cells = 0
    if left.exists() and right.exists():
        a, b = np.load(left, mmap_mode="r"), np.load(right, mmap_mode="r")
        assert a.shape == b.shape == (184800, 92) and a.dtype == b.dtype == np.float32
        for start in range(0, len(a), 2048):
            assert np.array_equal(a[start:start + 2048], b[start:start + 2048])
            cells += a[start:start + 2048].size
    print(json.dumps({"status": "PASS", "tracked_files": len(manifest["tracked_evidence"]),
                      "local_output_hashes": local_checks, "local_float32_cells_compared": cells,
                      "scale_status": scale["status"]}))


if __name__ == "__main__":
    main()
