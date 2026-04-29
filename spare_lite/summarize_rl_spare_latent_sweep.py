from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_seed_summaries(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("seed_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_verdict(summary: dict) -> dict:
    reward_delta = summary["delta"]["avg_reward"]
    hit_delta = summary["delta"]["avg_expert_hit"]
    prob_delta = summary["delta"]["avg_expert_prob"]
    baseline_hit = summary["baseline"]["avg_expert_hit_mean"]
    spare_hit = summary["spare"]["avg_expert_hit_mean"]
    has_greedy = "avg_greedy_expert_hit_mean" in summary["baseline"] and "avg_greedy_expert_hit_mean" in summary["spare"]
    greedy_hit_delta = summary["delta"].get("avg_greedy_expert_hit", 0.0)
    greedy_reward_delta = summary["delta"].get("avg_greedy_reward", 0.0)

    signal_present = spare_hit > 0.0 and summary["spare"]["avg_reward_mean"] > 0.0
    reward_improved = reward_delta > 0.0
    selection_not_worse = greedy_hit_delta >= 0.0 if has_greedy else hit_delta >= 0.0
    baseline_non_degenerate = baseline_hit > 0.0
    phase_a_ready = signal_present and reward_improved and selection_not_worse and baseline_non_degenerate

    reasons = []
    if signal_present:
        reasons.append("spare branch has nonzero reward and expert-hit signal")
    else:
        reasons.append("spare branch signal is still too weak")
    if reward_improved:
        reasons.append("mean reward improves over baseline")
    else:
        reasons.append("mean reward does not improve over baseline")
    if selection_not_worse:
        reasons.append("greedy expert-hit does not regress" if has_greedy else "expert-hit does not regress")
    else:
        reasons.append("greedy expert-hit regresses versus baseline" if has_greedy else "expert-hit regresses versus baseline")
    if baseline_non_degenerate:
        reasons.append("baseline is active enough for comparison")
    else:
        reasons.append("baseline remains too degenerate for a fair comparison")

    return {
        "phase_a_ready": phase_a_ready,
        "signal_present": signal_present,
        "reward_improved": reward_improved,
        "selection_not_worse": selection_not_worse,
        "baseline_non_degenerate": baseline_non_degenerate,
        "reward_delta": reward_delta,
        "expert_hit_delta": hit_delta,
        "expert_prob_delta": prob_delta,
        "greedy_reward_delta": greedy_reward_delta,
        "greedy_expert_hit_delta": greedy_hit_delta,
        "uses_greedy_selection_gate": has_greedy,
        "reasons": reasons,
    }


def summarize(rows: list[dict]) -> dict:
    baseline_rewards = [row["baseline"]["avg_reward"] for row in rows]
    spare_rewards = [row["spare"]["avg_reward"] for row in rows]
    baseline_hits = [row["baseline"]["avg_expert_hit"] for row in rows]
    spare_hits = [row["spare"]["avg_expert_hit"] for row in rows]
    baseline_probs = [row["baseline"]["avg_expert_prob"] for row in rows]
    spare_probs = [row["spare"]["avg_expert_prob"] for row in rows]
    has_greedy = all("avg_greedy_reward" in row["baseline"] and "avg_greedy_reward" in row["spare"] for row in rows)
    baseline_greedy_rewards = [row["baseline"]["avg_greedy_reward"] for row in rows] if has_greedy else []
    spare_greedy_rewards = [row["spare"]["avg_greedy_reward"] for row in rows] if has_greedy else []
    baseline_greedy_hits = [row["baseline"]["avg_greedy_expert_hit"] for row in rows] if has_greedy else []
    spare_greedy_hits = [row["spare"]["avg_greedy_expert_hit"] for row in rows] if has_greedy else []

    summary = {
        "num_seeds": len(rows),
        "baseline": {
            "avg_reward_mean": mean(baseline_rewards),
            "avg_expert_hit_mean": mean(baseline_hits),
            "avg_expert_prob_mean": mean(baseline_probs),
        },
        "spare": {
            "avg_reward_mean": mean(spare_rewards),
            "avg_expert_hit_mean": mean(spare_hits),
            "avg_expert_prob_mean": mean(spare_probs),
        },
        "delta": {
            "avg_reward": mean(spare_rewards) - mean(baseline_rewards),
            "avg_expert_hit": mean(spare_hits) - mean(baseline_hits),
            "avg_expert_prob": mean(spare_probs) - mean(baseline_probs),
        },
        "per_seed": rows,
    }
    if has_greedy:
        summary["baseline"]["avg_greedy_reward_mean"] = mean(baseline_greedy_rewards)
        summary["baseline"]["avg_greedy_expert_hit_mean"] = mean(baseline_greedy_hits)
        summary["spare"]["avg_greedy_reward_mean"] = mean(spare_greedy_rewards)
        summary["spare"]["avg_greedy_expert_hit_mean"] = mean(spare_greedy_hits)
        summary["delta"]["avg_greedy_reward"] = mean(spare_greedy_rewards) - mean(baseline_greedy_rewards)
        summary["delta"]["avg_greedy_expert_hit"] = mean(spare_greedy_hits) - mean(baseline_greedy_hits)
    summary["verdict"] = build_verdict(summary)
    return summary


def to_markdown(summary: dict) -> str:
    verdict = summary["verdict"]
    lines = [
        "# RL SpaRe Latent Sweep Summary",
        "",
        f"- seeds: `{summary['num_seeds']}`",
        f"- phase A ready: `{verdict['phase_a_ready']}`",
        f"- signal present: `{verdict['signal_present']}`",
        f"- reward improved: `{verdict['reward_improved']}`",
        f"- selection not worse: `{verdict['selection_not_worse']}`",
        f"- baseline non-degenerate: `{verdict['baseline_non_degenerate']}`",
        f"- baseline avg reward: `{summary['baseline']['avg_reward_mean']:.4f}`",
        f"- spare avg reward: `{summary['spare']['avg_reward_mean']:.4f}`",
        f"- reward delta: `{summary['delta']['avg_reward']:.4f}`",
        f"- baseline avg expert hit: `{summary['baseline']['avg_expert_hit_mean']:.4f}`",
        f"- spare avg expert hit: `{summary['spare']['avg_expert_hit_mean']:.4f}`",
        f"- expert hit delta: `{summary['delta']['avg_expert_hit']:.4f}`",
        f"- baseline avg expert prob: `{summary['baseline']['avg_expert_prob_mean']:.4f}`",
        f"- spare avg expert prob: `{summary['spare']['avg_expert_prob_mean']:.4f}`",
        f"- expert prob delta: `{summary['delta']['avg_expert_prob']:.4f}`",
    ]
    if "avg_greedy_reward_mean" in summary["baseline"]:
        lines.extend(
            [
                f"- baseline avg greedy reward: `{summary['baseline']['avg_greedy_reward_mean']:.4f}`",
                f"- spare avg greedy reward: `{summary['spare']['avg_greedy_reward_mean']:.4f}`",
                f"- greedy reward delta: `{summary['delta']['avg_greedy_reward']:.4f}`",
                f"- baseline avg greedy expert hit: `{summary['baseline']['avg_greedy_expert_hit_mean']:.4f}`",
                f"- spare avg greedy expert hit: `{summary['spare']['avg_greedy_expert_hit_mean']:.4f}`",
                f"- greedy expert hit delta: `{summary['delta']['avg_greedy_expert_hit']:.4f}`",
            ]
        )
    lines.extend([
        "",
        "## Verdict",
        "",
    ])
    for reason in verdict["reasons"]:
        lines.append(f"- {reason}")
    lines.extend([
        "",
        "## Per-seed",
        "",
    ])
    for row in summary["per_seed"]:
        seed = row["seed"]
        lines.extend(
            [
                f"### Seed {seed}",
                "",
                f"- baseline avg reward: `{row['baseline']['avg_reward']:.4f}`",
                f"- spare avg reward: `{row['spare']['avg_reward']:.4f}`",
                f"- baseline avg expert hit: `{row['baseline']['avg_expert_hit']:.4f}`",
                f"- spare avg expert hit: `{row['spare']['avg_expert_hit']:.4f}`",
                f"- baseline avg expert prob: `{row['baseline']['avg_expert_prob']:.4f}`",
                f"- spare avg expert prob: `{row['spare']['avg_expert_prob']:.4f}`",
            ]
        )
        if "avg_greedy_reward" in row["baseline"]:
            lines.extend(
                [
                    f"- baseline avg greedy reward: `{row['baseline']['avg_greedy_reward']:.4f}`",
                    f"- spare avg greedy reward: `{row['spare']['avg_greedy_reward']:.4f}`",
                    f"- baseline avg greedy expert hit: `{row['baseline']['avg_greedy_expert_hit']:.4f}`",
                    f"- spare avg greedy expert hit: `{row['spare']['avg_greedy_expert_hit']:.4f}`",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multi-seed rl_spare_latent_smoke outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    rows = load_seed_summaries(Path(args.input_dir))
    if not rows:
        raise SystemExit(f"No seed_*.json files found in {args.input_dir}")

    summary = summarize(rows)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[spare-lite-summary] wrote json summary to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(to_markdown(summary), encoding="utf-8")
        print(f"[spare-lite-summary] wrote markdown summary to {output_md}")

    print(
        "[spare-lite-summary] "
        f"seeds={summary['num_seeds']} "
        f"baseline_reward={summary['baseline']['avg_reward_mean']:.4f} "
        f"spare_reward={summary['spare']['avg_reward_mean']:.4f} "
        f"reward_delta={summary['delta']['avg_reward']:.4f} "
        f"phase_a_ready={summary['verdict']['phase_a_ready']}"
    )


if __name__ == "__main__":
    main()
