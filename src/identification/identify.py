"""Phase 3+4: Apply recalcitrance criterion and aggregate to token types."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table


# ---------------------------------------------------------------------------
# Phase 3: Classification
# ---------------------------------------------------------------------------

def classify_rock_tokens(
    loss_before: torch.Tensor,
    loss_after: torch.Tensor,
    tau_high_pct: float = 80,
    delta_pct: float = 20,
) -> torch.Tensor:
    """Classify token instances as Rock using the relative threshold.

    A token is Rock if:
      - loss_before >= tau_high  (was hard before training)
      - (loss_before - loss_after) < delta  (barely improved)

    Args:
        loss_before: (N,) KL with pre-OPD student
        loss_after:  (N,) KL with post-OPD student
        tau_high_pct: percentile of loss_before for the "hard" threshold
        delta_pct: percentile of improvement for the "barely improved" threshold

    Returns:
        (N,) boolean tensor — True for Rock token instances.
    """
    tau_high = torch.quantile(loss_before.float(), tau_high_pct / 100)
    improvement = loss_before - loss_after
    delta = torch.quantile(improvement.float(), delta_pct / 100)

    is_rock = (loss_before >= tau_high) & (improvement < delta)
    return is_rock


# ---------------------------------------------------------------------------
# Phase 4: Aggregation
# ---------------------------------------------------------------------------

def aggregate_to_types(
    token_ids: torch.Tensor,
    is_rock: torch.Tensor,
    min_frequency: int = 10,
    top_k: int = 100,
    loss_before: torch.Tensor | None = None,
    loss_after: torch.Tensor | None = None,
    teacher_entropy: torch.Tensor | None = None,
    student_entropy_before: torch.Tensor | None = None,
    student_entropy_after: torch.Tensor | None = None,
) -> list[dict]:
    """Aggregate per-instance Rock labels to per-type rankings.

    Groups by token_id, applies frequency filter, ranks by rock_rate.

    Returns list of dicts sorted by rock_rate descending, length <= top_k.
    """
    stats: dict[int, dict] = defaultdict(lambda: {
        "total": 0, "rock": 0,
        "loss_before_sum": 0.0, "loss_after_sum": 0.0,
        "teacher_entropy_sum": 0.0,
        "student_entropy_before_sum": 0.0, "student_entropy_after_sum": 0.0,
    })

    for i in range(len(token_ids)):
        tid = token_ids[i].item()
        s = stats[tid]
        s["total"] += 1
        if is_rock[i]:
            s["rock"] += 1
        if loss_before is not None:
            s["loss_before_sum"] += loss_before[i].item()
        if loss_after is not None:
            s["loss_after_sum"] += loss_after[i].item()
        if teacher_entropy is not None:
            s["teacher_entropy_sum"] += teacher_entropy[i].item()
        if student_entropy_before is not None:
            s["student_entropy_before_sum"] += student_entropy_before[i].item()
        if student_entropy_after is not None:
            s["student_entropy_after_sum"] += student_entropy_after[i].item()

    results = []
    for tid, s in stats.items():
        if s["total"] < min_frequency:
            continue
        n = s["total"]
        entry = {
            "token_id": tid,
            "frequency": n,
            "rock_count": s["rock"],
            "rock_rate": s["rock"] / n,
        }
        if loss_before is not None:
            entry["avg_loss_before"] = s["loss_before_sum"] / n
            entry["avg_loss_after"] = s["loss_after_sum"] / n
            entry["avg_improvement"] = entry["avg_loss_before"] - entry["avg_loss_after"]
        if teacher_entropy is not None:
            entry["avg_teacher_entropy"] = s["teacher_entropy_sum"] / n
        if student_entropy_before is not None:
            entry["avg_student_entropy_before"] = s["student_entropy_before_sum"] / n
        if student_entropy_after is not None:
            entry["avg_student_entropy_after"] = s["student_entropy_after_sum"] / n
        results.append(entry)

    results.sort(key=lambda x: x["rock_rate"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Phase 3+4 orchestration
# ---------------------------------------------------------------------------

def run_identification(config: dict, output_dir: Path) -> Path:
    """Load Phase 2 data, classify, aggregate, and save top-K Rock Tokens."""
    console = Console()
    output_json = output_dir / "rock_tokens.json"

    phase2_path = output_dir / "phase2_losses.pt"
    data = torch.load(phase2_path, weights_only=False)

    ic = config["identification"]
    console.print(
        f"Classifying with τ_high={ic['tau_high_percentile']}th pct, "
        f"δ={ic['delta_percentile']}th pct..."
    )

    is_rock = classify_rock_tokens(
        data["loss_before"], data["loss_after"],
        tau_high_pct=ic["tau_high_percentile"],
        delta_pct=ic["delta_percentile"],
    )

    n_total = len(is_rock)
    n_rock = is_rock.sum().item()
    rock_frac = n_rock / n_total
    console.print(f"Rock instances: {n_rock:,} / {n_total:,} ({rock_frac:.1%})")

    if rock_frac < 0.05 or rock_frac > 0.20:
        console.print(
            f"[bold yellow]WARNING: Rock fraction {rock_frac:.1%} is outside "
            f"expected 5-20% range. Consider adjusting thresholds.[/bold yellow]"
        )

    results = aggregate_to_types(
        data["token_ids"], is_rock,
        min_frequency=ic["min_frequency"],
        top_k=ic["top_k"],
        loss_before=data["loss_before"],
        loss_after=data["loss_after"],
        teacher_entropy=data["teacher_entropy"],
        student_entropy_before=data["student_entropy_before"],
        student_entropy_after=data["student_entropy_after"],
    )

    # Add rank and decode token strings
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student_onpolicy"], trust_remote_code=True
    )
    for rank, entry in enumerate(results, 1):
        entry["rank"] = rank
        entry["token_string"] = tokenizer.decode([entry["token_id"]])

    # Print table
    table = Table(title=f"Top {len(results)} Rock Tokens")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Token", style="cyan")
    table.add_column("ID", justify="right")
    table.add_column("Freq", justify="right")
    table.add_column("Rock#", justify="right")
    table.add_column("Rate", justify="right", style="bold")
    table.add_column("Avg KL₀", justify="right")
    table.add_column("Avg KL*", justify="right")

    for entry in results[:30]:
        table.add_row(
            str(entry["rank"]),
            repr(entry["token_string"]),
            str(entry["token_id"]),
            str(entry["frequency"]),
            str(entry["rock_count"]),
            f"{entry['rock_rate']:.3f}",
            f"{entry.get('avg_loss_before', 0):.3f}",
            f"{entry.get('avg_loss_after', 0):.3f}",
        )
    console.print(table)

    # Save JSON
    thresholds = {
        "tau_high_percentile": ic["tau_high_percentile"],
        "delta_percentile": ic["delta_percentile"],
        "tau_high_value": float(torch.quantile(
            data["loss_before"].float(), ic["tau_high_percentile"] / 100
        )),
        "delta_value": float(torch.quantile(
            (data["loss_before"] - data["loss_after"]).float(),
            ic["delta_percentile"] / 100,
        )),
    }

    output_data = {
        "metadata": {
            "models": config["models"],
            "total_token_positions": n_total,
            "rock_instances": n_rock,
            "rock_fraction": rock_frac,
            "thresholds": thresholds,
            "min_frequency": ic["min_frequency"],
            "top_k": ic["top_k"],
        },
        "rock_tokens": results,
    }

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    console.print(f"Saved {len(results)} Rock Tokens to {output_json}")

    csv_path = output_dir / "rock_tokens.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    console.print(f"Saved CSV to {csv_path}")

    return output_json
