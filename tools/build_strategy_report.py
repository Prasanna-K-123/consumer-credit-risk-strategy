"""Generate the retrospective strategy report from independently reviewed evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"scorecard": "WoE logistic", "challenger": "Boosting", "cart": "Shallow CART"}
COLORS = {"scorecard": "#176B87", "challenger": "#C45A27", "cart": "#637C45"}


def load(evidence: Path):
    review = json.loads((evidence / "independent_review.json").read_text())
    if review.get("status") != "PASS" or review.get("source_labels_reconciled") is not True:
        raise ValueError("Source-reconciled independent strategy review must pass")
    identities = review.get("input_sha256", {})
    required = {"summary.json", "predictions.csv", "policy_sensitivity.csv", "capacity.csv", "leaf_segments.csv", "tree.txt", "protocol.json", "receipt.json"}
    if not required.issubset(identities):
        raise ValueError("Independent review must bind the complete strategy evidence")
    for name in required:
        if hashlib.sha256((evidence / name).read_bytes()).hexdigest() != identities[name]:
            raise ValueError("Strategy evidence changed since independent review: " + name)
    summary = json.loads((evidence / "summary.json").read_text())
    policies = pd.read_csv(evidence / "policy_sensitivity.csv")
    leaves = pd.read_csv(evidence / "leaf_segments.csv")
    if set(policies.model) != set(NAMES) or len(policies) != 15:
        raise ValueError("All three models and five cost scenarios must be presented")
    for name in NAMES:
        if sorted(policies.loc[policies.model == name, "missed_default_cost"]) != [1, 2, 5, 10, 20]:
            raise ValueError("Declared cost-scenario coverage is incomplete")
    values = policies[["selected_threshold", "review_rate", "default_capture", "cost_units"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Strategy report requires finite quantitative evidence")
    return summary, policies, leaves, review


def build(evidence: Path, output: Path) -> list[Path]:
    summary, policies, leaves, review = load(Path(evidence))
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#142B3A", "text.color": "#142B3A"})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for name, label in NAMES.items():
        rows = policies[policies.model == name].sort_values("missed_default_cost")
        x = rows.missed_default_cost
        axes[0].plot(x, rows.cost_units / rows.n, "o-", color=COLORS[name], label=label)
        axes[1].plot(x, rows.review_rate, "o-", color=COLORS[name], label=label)
    baseline = policies[policies.model == "scorecard"].sort_values("missed_default_cost")
    axes[0].plot(baseline.missed_default_cost,
                 np.minimum(baseline.review_none_cost_units, baseline.review_all_cost_units) / baseline.n,
                 "--", color="#647480", label="Best constant benchmark")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
        axis.set_xlabel("Assumed missed-default cost / false-alarm cost")
        axis.grid(alpha=.18)
    axes[0].set(title="Historical cost under each stated assumption", ylabel="Error-cost units per evaluated client")
    axes[0].legend(fontsize=8, loc="best")
    axes[1].set(title="Review workload changes with the assumption", ylabel="Fraction of clients flagged", ylim=(-.02, 1.02))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.text(.015, .025, "Thresholds selected on policy-validation rows. Known Taiwan 2005 cohort; descriptive extension.\nCosts are assumed units, not lending losses, savings or measured intervention benefits.", fontsize=8, color="#526772")
    chart = figures / "policy_sensitivity.png"
    fig.savefig(chart, dpi=180, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)

    lines = ["# Retrospective card strategy: interpretable segments and cost sensitivity", "",
             "**Decision question.** How do readable customer segments and different assumed error costs change the workload of a review policy?", "",
             "**Use.** Compare the complete scenario set and explain the tradeoff. Confirm actual operating costs, capacity and intervention effects before selecting a business policy. A shallow tree provides readable segments; its simplicity does not establish superior risk prediction.", "",
             "This extension adds a fixed depth-three CART classifier in Python to the preserved scorecard and boosting research. The same 20 predictors and original four data roles are used: fitting on training rows, calibration on calibration rows, and threshold selection on policy-validation rows. Each ratio is selected independently on the declared 0.01–0.99 grid; the lowest threshold wins exact cost ties. Every declared result is reported.", "",
             "The historical cohort and its evaluation outcomes were already known when this extension was designed. These are retrospective comparisons, with no new untouched-test, US borrower-performance, monetary-profit, SAS or CHAID claim.", "",
             "## Predictive evidence", "", "| Model | ROC AUC | Average precision | Raw Brier | Calibrated Brier |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, label in NAMES.items():
        result = summary["models"][name]
        c, r = result["calibrated"], result["raw"]
        lines.append(f"| {label} | {c['auc']:.4f} | {c['average_precision']:.4f} | {r['brier']:.6f} | {c['brier']:.6f} |")
    lines += ["", "All evaluation results concern the same 6,001 historical customers. Extension results are descriptive point estimates; the original paired bootstrap intervals are not uncertainty intervals for this new tree or these additional scenarios.", "",
              "## Policy-cost scenarios", "", "![Review workload and error-cost sensitivity](figures/policy_sensitivity.png)", "",
              "| Model | FN:FP cost | Threshold | Reviewed | Defaults captured | Cost units | Best constant cost |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, label in NAMES.items():
        for row in policies[policies.model == name].sort_values("missed_default_cost").itertuples():
            constant = min(row.review_none_cost_units, row.review_all_cost_units)
            lines.append(f"| {label} | {row.missed_default_cost:g}:1 | {row.selected_threshold:.2f} | {row.reviewed:,} ({row.review_rate:.1%}) | {row.default_capture:.1%} | {row.cost_units:,.0f} | {constant:,.0f} |")
    lines += ["", "Cost = assumed FN weight × missed defaults + 1 × nondefault clients flagged. Review-all and review-none are transparent comparison policies. Costs at different ratios use different assumptions and cannot be read as changing realized bank losses. Higher review coverage also consumes more operational capacity. Ranking a client does not establish that a review changes the outcome.", "",
              "## Readable CART segments", "",
              f"The fixed tree has **{summary['tree']['leaves']} leaves**, depth **{summary['tree']['depth']}**, and at least **{summary['tree']['smallest_training_leaf']} training rows per leaf**. Splits minimize training Gini impurity; calibration and evaluation outcomes do not select the tree structure. PAY status codes are split numerically and must be interpreted using the original codebook. Age remains a predictor; no fairness or regulatory certification follows.", "",
              "| Leaf | Training clients | Training default rate | Evaluation clients | Evaluation default rate | Rule |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for leaf in sorted(leaves.leaf_id.unique()):
        training = leaves[(leaves.partition == "train") & (leaves.leaf_id == leaf)].iloc[0]
        test = leaves[(leaves.partition == "test") & (leaves.leaf_id == leaf)].iloc[0]
        def rate(row):
            return f"{row.default_rate:.1%}" if row.n else "undefined (empty)"
        lines.append(f"| {leaf} | {int(training.n):,} | {rate(training)} | {int(test.n):,} | {rate(test)} | `{training.rule}` |")
    lines += ["", "Leaf rates describe observed concentration in a single historical cohort. Compare role-specific support and errors before treating a segment as stable. Raw and calibrated probabilities, all four role counts and exact paths are supplied in `reference/card_strategy/leaf_segments.csv`; the complete tree is in `tree.txt`.", "",
              "## Independent checks and reproduction", "",
              "A separate SQLite/direct-arithmetic implementation rechecks scenario selection, evaluation accounting, capacity ordering, metrics and leaf support. The report refuses missing source reconciliation, a failed review or evidence changed since that review. The fixed original models' evaluation probabilities are also reconciled to the preserved historical predictions.", "",
              "Run `python tools/reproduce.py --output work/current` for the complete chain. For a standalone extension, run `python -m card_strategy --source .cache/uci_card_clients.xls --output work/strategy`, then `python tools/audit_strategy.py --evidence work/strategy --source .cache/uci_card_clients.xls --output work/strategy/independent_review.json`.", "",
              "Evidence: [protocol](../reference/card_strategy/protocol.json), [scenario table](../reference/card_strategy/policy_sensitivity.csv), [independent review](../reference/card_strategy/independent_review.json), [source/method limits](../docs/CARD_METHODOLOGY.md).", ""]
    report = output / "STRATEGY_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return [report, chart]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "reference/card_strategy")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory() as temporary:
            fresh = Path(temporary)
            for path in build(args.evidence, fresh):
                relative = path.relative_to(fresh)
                if not (args.output / relative).is_file() or path.read_bytes() != (args.output / relative).read_bytes():
                    raise ValueError("Stale strategy presentation: " + str(relative))
        print("Strategy report and chart match independently reviewed evidence")
    else:
        for path in build(args.evidence, args.output):
            print(path)


if __name__ == "__main__":
    main()
