from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_brief(summary: dict, decision: dict) -> str:
    verdict = summary["verdict"]
    lines = [
        "# SpaRe-lite Result Brief",
        "",
        "## Current status",
        "",
        f"- phase A ready: `{verdict['phase_a_ready']}`",
        f"- current phase complete: `{decision['current_phase_complete']}`",
        f"- next phase: `{decision['next_phase']}`",
        "",
        "## What we can claim now",
        "",
        "- The supervised SpaRe-lite path runs on real checkpoints and real data subsets.",
        "- The lightweight latent-validation path also runs end-to-end.",
        f"- In the current sweep summary, baseline average reward is `{summary['baseline']['avg_reward_mean']:.4f}` and SpaRe average reward is `{summary['spare']['avg_reward_mean']:.4f}`.",
        f"- The current reward delta is `{summary['delta']['avg_reward']:.4f}`.",
        "",
        "## What we should not overclaim",
        "",
        "- This is still a lightweight validation path, not a final online RL benchmark result.",
        "- The current result should be framed as a controlled diagnostic comparison, not a full task-level success-rate claim.",
        "",
        "## Why the current stage may still be incomplete",
        "",
    ]
    for reason in verdict["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Immediate next action",
            "",
            f"- {decision['instruction']}",
            "",
            "## Suggested presenter framing",
            "",
            "- Right now, the main contribution of this stage is that the SpaRe-lite validation path is executable, measurable, and structured enough to compare `R1` and `R1 + lambda R2` in a controlled setup.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a concise result brief from sweep artifacts.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    decision = json.loads(Path(args.decision_json).read_text(encoding="utf-8"))
    brief = build_brief(summary, decision)

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(brief, encoding="utf-8")
    print(f"[spare-lite-brief] wrote result brief to {output_path}")


if __name__ == "__main__":
    main()
