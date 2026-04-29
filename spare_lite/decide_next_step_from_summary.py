from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_decision(summary: dict) -> dict:
    verdict = summary["verdict"]
    if verdict["phase_a_ready"]:
        return {
            "current_phase_complete": True,
            "next_phase": "phase_b_reproducibility_and_result_packaging",
            "instruction": (
                "Phase A looks ready. Move to phase B: keep the current validation "
                "setup fixed, collect one cleaner reproducibility batch, and start "
                "writing the concise result framing for the poster/report."
            ),
        }

    return {
        "current_phase_complete": False,
        "next_phase": "phase_a_validation_stabilization",
        "instruction": (
            "Phase A is not ready yet. Continue improving the validation setup: "
            "keep batch_size=4 as the default quick check, run the multi-seed sweep, "
            "and refine the setup only if the summary verdict remains negative."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide the next SpaRe-lite step from a sweep summary.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decision = build_decision(summary)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        print(f"[spare-lite-decision] wrote decision to {output_path}")

    print(
        "[spare-lite-decision] "
        f"current_phase_complete={decision['current_phase_complete']} "
        f"next_phase={decision['next_phase']}"
    )
    print(decision["instruction"])


if __name__ == "__main__":
    main()
