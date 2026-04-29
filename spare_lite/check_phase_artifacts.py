from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PHASE_A_FILES = [
    "summary.json",
    "summary.md",
    "decision.json",
    "result_brief.md",
]


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_status(output_dir: Path) -> dict:
    existing = sorted(p.name for p in output_dir.iterdir()) if output_dir.exists() else []
    missing = [name for name in REQUIRED_PHASE_A_FILES if not (output_dir / name).exists()]

    summary = load_json_if_exists(output_dir / "summary.json")
    decision = load_json_if_exists(output_dir / "decision.json")

    phase_a_ready = None
    next_phase = None
    instruction = None
    if summary is not None:
        phase_a_ready = summary.get("verdict", {}).get("phase_a_ready")
    if decision is not None:
        next_phase = decision.get("next_phase")
        instruction = decision.get("instruction")

    if missing:
        state = "incomplete_artifacts"
    elif phase_a_ready is True:
        state = "phase_a_complete"
    elif phase_a_ready is False:
        state = "phase_a_incomplete"
    else:
        state = "unknown"

    return {
        "output_dir": str(output_dir),
        "existing_files": existing,
        "missing_files": missing,
        "state": state,
        "phase_a_ready": phase_a_ready,
        "next_phase": next_phase,
        "instruction": instruction,
    }


def to_markdown(status: dict) -> str:
    lines = [
        "# SpaRe-lite Artifact Status",
        "",
        f"- output dir: `{status['output_dir']}`",
        f"- state: `{status['state']}`",
        f"- phase A ready: `{status['phase_a_ready']}`",
        f"- next phase: `{status['next_phase']}`",
        "",
        "## Missing files",
        "",
    ]
    if status["missing_files"]:
        for name in status["missing_files"]:
            lines.append(f"- `{name}`")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Existing files",
        "",
    ])
    if status["existing_files"]:
        for name in status["existing_files"]:
            lines.append(f"- `{name}`")
    else:
        lines.append("- none")

    if status["instruction"]:
        lines.extend([
            "",
            "## Next instruction",
            "",
            f"- {status['instruction']}",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether SpaRe-lite phase artifacts are complete.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    status = build_status(output_dir)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[spare-lite-artifacts] wrote json status to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(to_markdown(status), encoding="utf-8")
        print(f"[spare-lite-artifacts] wrote markdown status to {output_md}")

    print(
        "[spare-lite-artifacts] "
        f"state={status['state']} "
        f"phase_a_ready={status['phase_a_ready']} "
        f"missing={len(status['missing_files'])}"
    )


if __name__ == "__main__":
    main()
