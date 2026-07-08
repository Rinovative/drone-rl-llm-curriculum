#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/render_run_gifs.sh <run-manifest-path> [output-dir]" >&2
  exit 2
fi

RUN_MANIFEST="$1"
OUTPUT_DIR="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$RUN_MANIFEST" ]]; then
  echo "run manifest not found: $RUN_MANIFEST" >&2
  exit 2
fi

"$PYTHON_BIN" - "$RUN_MANIFEST" "$OUTPUT_DIR" <<'PYRENDER'
from pathlib import Path
import csv
import json
import sys

manifest_path = Path(sys.argv[1]).resolve(strict=False)
output_arg = sys.argv[2]
run_root = manifest_path.parent
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
output_dir = Path(output_arg) if output_arg else run_root / "evaluations" / "render_status"
output_dir.mkdir(parents=True, exist_ok=True)
artifacts = []
for suffix in ("*.gif", "*.mp4"):
    for path in sorted(run_root.glob(f"**/{suffix}")):
        artifacts.append(
            {
                "path": str(path),
                "path_relative": str(path.relative_to(run_root)) if path.is_relative_to(run_root) else str(path),
                "kind": path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size,
            }
        )
status = "found" if artifacts else "no_render_artifacts_found"
report = {
    "run_name": manifest.get("run_name"),
    "run_kind": manifest.get("run_kind"),
    "status": status,
    "artifact_count": len(artifacts),
    "artifacts": artifacts,
    "note": "Evaluation suites render deterministic GIFs when simulator rendering succeeds; this script reports discovered artifacts without deleting or overwriting runs.",
}
json_path = output_dir / "render_status.json"
tsv_path = output_dir / "render_status.tsv"
json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with tsv_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = ["kind", "path", "path_relative", "size_bytes"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(artifacts)
print(json.dumps({"status": status, "artifact_count": len(artifacts), "report_path": str(json_path), "summary_path": str(tsv_path)}, indent=2, sort_keys=True))
PYRENDER
