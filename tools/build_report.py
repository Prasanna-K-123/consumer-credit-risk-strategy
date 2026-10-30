"""Build the research presentation from executed evidence, without fitting models.

Run from the repository root. ``--check`` builds in a temporary directory and
rejects stale Markdown or figures. No network access is used by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NAMES = {"scorecard": "WoE logistic", "challenger": "Boosting challenger"}
COLORS = {"scorecard": "#176B87", "challenger": "#C45A27"}
INK, GRID = "#142B3A", "#D9E2E8"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON value: " + value)))
    if not isinstance(value, dict):
        raise ValueError(f"Object required: {path}")
    return value


def number(value, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Finite numeric evidence required: " + label)
    return float(value)


def validate_evidence(card: dict, calibration: pd.DataFrame, queues: pd.DataFrame, audit: dict) -> None:
    """Require executed, internally consistent evidence before making figures."""
    if audit.get("status") != "PASS":
        raise ValueError("Independent card audit must pass before report generation")
    if not audit.get("controls") or any(row.get("violations") != 0 for row in audit["controls"]):
        raise ValueError("Independent card controls are incomplete or have violations")
    if audit.get("queue_reference_checked") is not True:
        raise ValueError("Independent queue reconciliation must pass before report generation")
    expected_rows = card["partitions"]["test"]["rows"]
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows <= 0:
        raise ValueError("Positive integral test population required")
    required = {"model", "n", "mean_probability", "default_rate"}
    if not required.issubset(calibration.columns) or calibration.empty:
        raise ValueError("Calibration-bin evidence missing")
    if not {"model", "capacity_fraction", "flagged", "capture_rate", "precision"}.issubset(queues.columns) or queues.empty:
        raise ValueError("Decision-queue evidence missing")
    if set(calibration.model) != set(NAMES) or set(queues.model) != set(NAMES):
        raise ValueError("Exact model identities required")
    for name in NAMES:
        result = card["models"][name]
        independent = audit["independent_metrics"][name]
        if independent["rows"] != expected_rows:
            raise ValueError("Independent population disagrees with study")
        for key in ("auc", "brier", "log_loss"):
            value = number(result[key], name + "." + key)
            if not math.isclose(value, number(independent[key], "independent " + key), rel_tol=1e-9, abs_tol=1e-11):
                raise ValueError("Independent metric disagreement: " + name + "." + key)
        group = calibration[calibration.model == name]
        if not np.isfinite(group.n).all() or (group.n < 0).any() or not (group.n == np.floor(group.n)).all():
            raise ValueError("Nonnegative integral calibration-bin counts required")
        active_bins = group[group.n > 0]
        empty_bins = group[group.n == 0]
        if not empty_bins[["mean_probability", "default_rate"]].isna().all().all():
            raise ValueError("Empty calibration bins must retain undefined rates")
        if group.n.sum() != expected_rows or not np.isfinite(active_bins[["mean_probability", "default_rate"]].to_numpy()).all():
            raise ValueError("Calibration population or numeric evidence invalid")
        if ((active_bins[["mean_probability", "default_rate"]] < 0) | (active_bins[["mean_probability", "default_rate"]] > 1)).any().any():
            raise ValueError("Calibration probabilities outside [0,1]")
        q = queues[queues.model == name]
        if q.capacity_fraction.duplicated().any() or not np.isfinite(q.capacity_fraction).all():
            raise ValueError("Invalid or duplicate capacities")
        if ((q.capacity_fraction < 0) | (q.capacity_fraction > 1)).any():
            raise ValueError("Capacity outside [0,1]")
        if not np.array_equal(q.flagged.to_numpy(), np.floor(q.capacity_fraction.to_numpy() * expected_rows).astype(int)):
            raise ValueError("Queue counts disagree with stated capacity rule")
        for key in ("capture_rate", "precision"):
            active = q[q.flagged > 0][key]
            if not np.isfinite(active).all() or ((active < 0) | (active > 1)).any():
                raise ValueError("Invalid active-queue evidence: " + key)


def theme() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": GRID, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.hashsalt": "consumer-credit-risk-strategy-v1",
    })


def finish(fig, path: Path, caption: str) -> None:
    fig.text(0.01, 0.012, caption, fontsize=8, color="#526772", va="bottom", wrap=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)


def model_figure(card: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    x = np.arange(2)
    for i, name in enumerate(NAMES):
        axes[0].bar(i, card["models"][name]["auc"], width=.56, color=COLORS[name])
        axes[0].text(i, card["models"][name]["auc"] + .012,
                     f'{card["models"][name]["auc"]:.4f}', ha="center", weight="bold")
    axes[0].set(ylim=(0, 1), xticks=x, xticklabels=list(NAMES.values()), ylabel="ROC AUC", title="ROC AUC on the same 6,001 clients")
    raw = [card["models"][name]["raw_test_brier"] for name in NAMES]
    calibrated = [card["models"][name]["brier"] for name in NAMES]
    for i in range(2):
        axes[1].plot([raw[i], calibrated[i]], [i, i], color="#A7BCC6", linewidth=3, zorder=1)
        axes[1].text(raw[i], i-.20, f"{raw[i]:.6f}", ha="center", fontsize=9)
        axes[1].text(calibrated[i], i+.18, f"{calibrated[i]:.6f}", ha="center", fontsize=9)
    axes[1].scatter(raw, x, s=55, label="Raw score", color="#607D8B", zorder=3)
    axes[1].scatter(calibrated, x, s=55, label="Calibrated score", color="#176B87", zorder=3)
    axes[1].set(ylim=(-.65, 1.5), yticks=x, yticklabels=list(NAMES.values()), xlabel="Brier score · lower is better", title="Calibration did not improve test Brier")
    values = raw + calibrated
    axes[1].set_xlim(min(values)-.0005, max(values)+.0005)
    axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend(frameon=False, loc="lower left", ncol=2, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .10, 1, .98), w_pad=3)
    finish(fig, path, "UCI Taiwan 2005 snapshot · historical customer-group holdout; this replay adds no new unseen population.")


def calibration_figure(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), sharex=True, sharey=True)
    for ax, name in zip(axes, NAMES):
        group = frame[(frame.model == name) & (frame.n > 0)]
        ax.plot([0, 1], [0, 1], color="#8D9CA4", linestyle="--", linewidth=1, label="Equal predicted / observed")
        ax.plot(group.mean_probability, group.default_rate, color=COLORS[name], linewidth=1.5)
        ax.scatter(group.mean_probability, group.default_rate, s=25 + 4*np.sqrt(group.n),
                   color=COLORS[name], edgecolor="white", linewidth=.7, zorder=3)
        ax.set(title=NAMES[name], xlabel="Mean calibrated probability", xlim=(0, 1), ylim=(0, 1))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(color=GRID, linewidth=.6)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Observed next-month default rate")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .98))
    finish(fig, path, "Fixed 0.1-wide probability bins; marker area increases with client count. Small high-risk bins are noisy diagnostics.")


def queue_figure(frame: pd.DataFrame, prevalence: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    for name in NAMES:
        group = frame[(frame.model == name) & (frame.capacity_fraction > 0) & (frame.capacity_fraction <= .40)].sort_values("capacity_fraction")
        for ax, key in zip(axes, ("capture_rate", "precision")):
            ax.plot(group.capacity_fraction, group[key], "o-", color=COLORS[name], label=NAMES[name], linewidth=1.8, markersize=5)
    axes[0].plot([0, .4], [0, .4], color="#8D9CA4", linestyle="--", label="Random queue: expected capture")
    axes[1].axhline(prevalence, color="#8D9CA4", linestyle="--", label="Full-sample default prevalence")
    axes[0].set(title="How much observed default is captured?", ylabel="Share of defaults in review queue")
    axes[1].set(title="How concentrated is the review queue?", ylabel="Observed default share in queue")
    for ax in axes:
        ax.set(xlabel="Review capacity · share of 6,001 clients", xlim=(0, .42), ylim=(0, 1))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(color=GRID, linewidth=.6)
        ax.legend(frameon=False, fontsize=8, loc="lower right" if ax is axes[0] else "upper right")
    fig.tight_layout(rect=(0, .12, 1, .98), w_pad=2.3)
    finish(fig, path, "Postpublication diagnostic · descending calibrated risk, source ID breaks ties; no effect of reviewing a client is identified.")


def market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept the pack's stable wide table and enforce the plotted fields."""
    period = "quarter" if "quarter" in frame else "YRQTR" if "YRQTR" in frame else None
    required = ["dpd90_balances_pct", "dpd90_accounts_pct", "nco_annualized_pct", "balances_bn", "revolving_balances_bn"]
    if period is None or not set(required).issubset(frame):
        raise ValueError("US market quarterly fields missing")
    result = frame.copy()
    result["quarter"] = result[period]
    if result.quarter.duplicated().any() or not result.quarter.astype(str).str.fullmatch(r"\d{4}Q[1-4]").all():
        raise ValueError("Unique YYYYQn market quarters required")
    for key in required:
        result[key] = pd.to_numeric(result[key], errors="raise")
        if not np.isfinite(result[key]).all():
            raise ValueError("Nonfinite plotted US market metric: " + key)
    return result.sort_values("quarter")


def market_figure(frame: pd.DataFrame, path: Path) -> None:
    frame = market_frame(frame)
    x = np.arange(len(frame))
    tick = [i for i, quarter in enumerate(frame.quarter) if quarter.endswith("Q1") and int(quarter[:4]) % 2 == 0 and i <= len(frame)-5]
    if len(frame)-1 not in tick:
        tick.append(len(frame)-1)
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 8.4), sharex=True)
    axes[0].plot(x, frame.dpd90_balances_pct, color="#176B87", linewidth=1.8, label="Balance based")
    axes[0].plot(x, frame.dpd90_accounts_pct, color="#C45A27", linewidth=1.8, label="Account based")
    axes[0].set(title="90+ days past due: denominator changes interpretation", ylabel="Percent")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].plot(x, frame.nco_annualized_pct, color="#7A4C91", linewidth=1.8)
    axes[1].set(title="Net charge-offs: annualized quarterly flow rate", ylabel="Annualized percent")
    axes[2].plot(x, frame.balances_bn, color="#176B87", linewidth=1.8, label="All balances")
    axes[2].plot(x, frame.revolving_balances_bn, color="#C45A27", linewidth=1.8, label="Revolving balances")
    axes[2].set(title="Exposure: total and revolving balances", ylabel="USD billions", xticks=tick, xticklabels=[frame.quarter.iloc[i] for i in tick])
    axes[2].legend(frameon=False, ncol=2, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=.6)
        ax.set_ylim(bottom=0)
        ax.margins(x=.01)
    fig.tight_layout(rect=(0, .08, 1, 1), h_pad=2.2)
    finish(fig, path, "Philadelphia Fed FR Y-14M public large-bank aggregates · 2026Q2 source release, as retrieved; historical observations can be revised.")


def origination_figure(frame: pd.DataFrame, path: Path) -> None:
    frame = market_frame(frame)
    keys = ("new_sub660_accounts_pct", "new_sub660_commitments_pct", "new_sub660_avg_limit_usd", "new_ge660_avg_limit_usd")
    if not set(keys).issubset(frame) or not np.isfinite(frame[list(keys)].to_numpy()).all():
        raise ValueError("Origination mix and implied line-size evidence required")
    x = np.arange(len(frame))
    tick = [i for i, quarter in enumerate(frame.quarter) if quarter.endswith("Q1") and int(quarter[:4]) % 2 == 0 and i <= len(frame)-5]
    tick.append(len(frame)-1)
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.4), sharex=True)
    axes[0].plot(x, frame.new_sub660_accounts_pct, color="#176B87", label="Share of new accounts", linewidth=1.8)
    axes[0].plot(x, frame.new_sub660_commitments_pct, color="#C45A27", label="Share of new commitments", linewidth=1.8)
    axes[0].set(title="Below-660 origination mix: accounts and commitments", ylabel="Percent of quarterly originations")
    axes[1].plot(x, frame.new_sub660_avg_limit_usd, color="#176B87", label="Original score below 660", linewidth=1.8)
    axes[1].plot(x, frame.new_ge660_avg_limit_usd, color="#C45A27", label="Original score 660 or above", linewidth=1.8)
    axes[1].set(title="Implied average initial credit limit", ylabel="USD per new account", xticks=tick, xticklabels=[frame.quarter.iloc[i] for i in tick])
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=.6)
        ax.set_ylim(bottom=0)
        ax.margins(x=.01)
        ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout(rect=(0, .10, 1, 1), h_pad=2.2)
    finish(fig, path, "Philadelphia Fed FR Y-14M · implied averages use rounded aggregate commitments/account shares; they are not reported medians or future default rates.")


def validate_market(frame: pd.DataFrame, report: dict) -> None:
    checked = market_frame(frame)
    coverage = report.get("coverage", {})
    if coverage.get("quarters") != len(checked) or coverage.get("first_quarter") != checked.quarter.iloc[0] or coverage.get("last_quarter") != checked.quarter.iloc[-1]:
        raise ValueError("Market report coverage disagrees with quarterly evidence")
    if report.get("quality", {}).get("errors") != 0 or report.get("quality", {}).get("checks", 0) <= 0:
        raise ValueError("Market quality controls must pass")
    latest = checked.iloc[-1]
    for key, value in report.get("latest", {}).items():
        if key == "quarter":
            if value != latest.quarter:
                raise ValueError("Market latest quarter mismatch")
        elif key in checked:
            if not math.isclose(number(value, key), number(latest[key], key), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Market latest metric mismatch: " + key)
    indicators = report.get("year_on_year_indicators")
    if not indicators:
        raise ValueError("Market year-on-year report evidence required")
    prior_quarter = str(int(latest.quarter[:4])-1) + latest.quarter[4:]
    prior = checked[checked.quarter == prior_quarter]
    if len(prior) != 1:
        raise ValueError("Market year-earlier quarter missing")
    for item in indicators:
        key = item["metric"]
        for actual, expected in ((item["current"], latest[key]), (item["year_earlier"], prior.iloc[0][key]), (item["yoy_change_pp"], latest[key] - prior.iloc[0][key])):
            if not math.isclose(number(actual, key), number(expected, key), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Market year-on-year indicator mismatch: " + key)


def fmt(value, digits=4) -> str:
    return f"{number(value, 'reported value'):,.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(row) + " |" for row in rows])


def build_markdown(card, calibration, queues, audit, market, market_report, identities):
    test = card["partitions"]["test"]
    prevalence = test["defaults"] / test["rows"]
    training_prevalence = card["partitions"]["train"]["default_rate"]
    constant_brier = prevalence * (1-training_prevalence)**2 + (1-prevalence) * training_prevalence**2
    latest = market_frame(market).iloc[-1]
    model_rows = []
    for name, label in NAMES.items():
        m = card["models"][name]
        independent = audit["independent_metrics"][name]
        model_rows.append([label, fmt(m["auc"]), fmt(independent["average_precision"]), fmt(m["brier"]), fmt(m["raw_test_brier"]), fmt(m["log_loss"]), fmt(m["threshold"], 2), f'{m["cost_units"]:,}'])
    model_table = table(["Model", "ROC AUC", "Average precision", "Brier calibrated", "Brier raw", "Log loss", "Frozen threshold", "Assumed cost units"], model_rows)
    uncertainty_rows = []
    for key, label in (("auc", "ROC AUC"), ("brier", "Brier"), ("cost_per_client", "Assumed cost / client")):
        result = card["paired_uncertainty"][key]
        uncertainty_rows.append([label, fmt(result["challenger_minus_scorecard"], 6), f'[{fmt(result["ci95"][0], 6)}, {fmt(result["ci95"][1], 6)}]'])
    uncertainty_table = table(["Challenger minus scorecard", "Observed difference", "95% paired group-bootstrap interval"], uncertainty_rows)
    queue_rows = []
    for _, q in queues[queues.capacity_fraction.isin([.05, .10, .20])].sort_values(["capacity_fraction", "model"]).iterrows():
        queue_rows.append([NAMES[q.model], f"{q.capacity_fraction:.0%}", f"{int(q.flagged):,}", f"{int(q.defaults_flagged):,}", f"{q.capture_rate:.1%}", f"{q.precision:.1%}"])
    queue_table = table(["Model", "Capacity", "Clients flagged", "Defaults flagged", "Default capture", "Queue default rate"], queue_rows)
    partition_table = table(["Role", "Clients", "Feature groups", "Defaults"], [[key.replace("_", " ").title(), f'{value["rows"]:,}', f'{value["groups"]:,}', f'{value["defaults"]:,}'] for key, value in card["partitions"].items()])
    controls_table = table(["Executed control", "Violations"], [[str(row["control"]), str(row["violations"])] for row in audit["controls"]])
    latest_table = table(["Metric", f'{latest["quarter"]} value', "Unit / meaning"], [
        ["Total balances", fmt(latest["balances_bn"], 2), "USD billions · quarter-end stock"],
        ["Revolving balances", fmt(latest["revolving_balances_bn"], 2), "USD billions · subset of balances"],
        ["90+ DPD, balances", fmt(latest["dpd90_balances_pct"], 2), "Percent of balances"],
        ["90+ DPD, accounts", fmt(latest["dpd90_accounts_pct"], 2), "Percent of accounts"],
        ["Net charge-off rate", fmt(latest["nco_annualized_pct"], 2), "Annualized percent · quarterly flow rate"],
    ])
    market_yoy = table(["Market indicator", "Current (%)", "Year earlier (%)", "Change (percentage points)"], [[item["label"], fmt(item["current"], 2), fmt(item["year_earlier"], 2), f'{number(item["yoy_change_pp"], item["metric"]):+.2f}'] for item in market_report["year_on_year_indicators"]])
    source_rows = [[name, f'`{identity}`'] for name, identity in identities.items()]
    research = f"""# Consumer credit risk: models, review queues and market evidence

This study asks how a risk reviewer can compare an interpretable scorecard with a flexible challenger, quantify the workload of a fixed review policy, and reconcile the result independently. A separate market pack describes US consumer-bankcard conditions using public Federal Reserve aggregates.

## Decision

The challenger has higher historical ROC AUC; the scorecard has slightly higher average precision. Probability-quality and assumed operating-cost gains remain uncertain. Keep both models as research benchmarks and investigate calibration and segment behavior before proposing a new policy. The US market pack supplies exposure and delinquency context; its aggregates are never joined to Taiwanese clients or used as evidence of US borrower-model performance.

## 1. Existing-card customer risk: historical Taiwan snapshot

UCI's 30,000 existing credit-card customers supply the same 20 model predictors for each model. Identical model-input vectors stay in one partition. ID, sex, education and marital status do not enter the models; sex remains available for diagnostic comparison. Age is a model predictor. The final historical evaluation contains **{test['rows']:,} clients and {test['defaults']:,} defaults ({prevalence:.2%})**.

{partition_table}

The scorecard learns five-quantile weight-of-evidence transforms on training data. The challenger is the historical fixed histogram-gradient-boosting model. Each uses the same training, calibration, policy-validation and test clients. Separate regularized logistic calibration and policy-validation choices remain frozen. A review threshold minimizes the stated scenario cost **5 × missed defaults + nondefault clients flagged** on policy-validation data.

{model_table}

A constant score equal to training default prevalence (**{training_prevalence:.2%}**) has test Brier **{constant_brier:.5f}**, calculated from the frozen train/test counts. Under the stated cost assumption, flagging nobody costs **{card['constant_policies']['accept_all_cost_units']:,} units** and flagging everybody costs **{card['constant_policies']['reject_all_cost_units']:,} units**. The model policies improve on these simple reference decisions in this historical sample; that comparison does not measure an intervention benefit.

![Ranking and raw-versus-calibrated probability quality on the historical client holdout.](figures/model_comparison.png)

Calibration increases test Brier for both historical models. This observation is retained as a finding; these known outcomes are not used to select a replacement calibration. The scorecard has slightly higher average precision despite the challenger's higher ROC AUC. Average precision should be read against the **{prevalence:.2%}** sample default prevalence.

{uncertainty_table}

Intervals use **{card['paired_group_bootstrap_replicates']:,} paired model-input-group resamples** with fitted models, calibration and thresholds fixed. They do not include fitting or model-selection uncertainty. The cost interval includes zero. ROC improvement alone does not establish a useful policy improvement.

![Calibration by fixed probability bins, with client-count-sensitive markers.](figures/calibration.png)

## 2. Review capacity: what is actually measured

{queue_table}

![Observed default capture and concentration as review capacity changes.](figures/review_capacity.png)

These postpublication queues sort calibrated probabilities descending and break ties by source ID. They quantify observed default concentration and nondefault review workload. They do not identify whether reviewing a client prevents default, reduces loss, or increases profit. Monetary loss severity, recoveries and treatment effects are not observed here. The original threshold-policy cost comparison is an assumed 5:1 error tradeoff, not a currency return.

## 3. Independent SQL and numerical challenge

SQLite statements reproduce population, threshold confusion accounting, calibration bins, segment summaries and capacity queues. Separate rank-grouping calculations verify ROC AUC and average precision; direct arithmetic verifies Brier and log loss. The audit rejects missing, extra or duplicate identities, inconsistent partition labels and invalid probabilities before calculation.

{controls_table}

The review status is **{audit['status']}**. This is a second software implementation inside one project, not an outside validator's approval. Query files and CSV evidence are committed; the reproduction commands also generate a queryable SQLite database locally under `reference/card_clients/sql_review/`.

## 4. US large-bank card conditions: a separate population

The Philadelphia Fed market pack contains **{len(market):,} quarterly observations per source table**, from **{market_frame(market).quarter.iloc[0]} to {latest['quarter']}**. Balances, commitments and accounts are stocks; originations and purchases are quarterly flows. Net charge-offs are annualized. Balance-based and account-based delinquency have different denominators.

{latest_table}

![US bankcard delinquency, annualized net charge-offs and balances from the current source release.](figures/us_card_market.png)

{market_yoy}

![Below-660 origination shares and implied average initial credit limits.](figures/origination_mix.png)

Below-660 borrowers account for **{latest['new_sub660_accounts_pct']:.2f}%** of new accounts and **{latest['new_sub660_commitments_pct']:.2f}%** of new commitments in the latest quarter. The implied average initial line is about **USD {latest['new_sub660_avg_limit_usd']:,.0f}** for that group. These rounded-aggregate line-size diagnostics motivate investigation of mix and exposure; they do not measure subsequent defaults.

The pack preserves missing source values, validates headers, units, quarter keys and source footers, checks balance/account relationships, and generates year-over-year and quarter-over-quarter comparisons. A pinned release reproduces this retrieved version of revised history. It does not establish the exact information available at every historical publication date. Public aggregate reporting does not disclose the account microdata required for a US borrower-level credit model.

## 5. Provenance, reproduction and scope

The original card protocol, implementation and results are preserved under `reference/card_clients/historical/`. Their first recorded commit contains protocol and outcomes together; this is a historical recorded protocol, not independently certified preregistration. The current replay and added diagnostics are postpublication work on known outcomes. The UCI source is credited to I-Cheng Yeh under CC BY 4.0; Philadelphia Fed source terms are separate from the software license.

See `docs/CARD_METHODOLOGY.md`, `docs/CARD_HISTORY.md`, `docs/SQL_CONTROLS.md` and `docs/MARKET_METHODOLOGY.md` for definitions and full reproduction instructions. This presentation is generated from the executed replay, independent review and market pack. Input identities:

{table(['Input', 'SHA256'], source_rows)}

The data support historical existing-card risk ranking and a reproducible market report. Deployment, acquisition underwriting, realized profitability, independent institutional approval, and CECL/CCAR compliance require evidence this study does not contain.
"""
    ten = queues[(queues.model == "challenger") & np.isclose(queues.capacity_fraction, .10)]
    if len(ten) != 1:
        raise ValueError("Exactly one challenger 10% capacity result required")
    q = ten.iloc[0]
    memo = f"""# Decision memo: consumer-card risk evidence

**Question.** Does a flexible risk model justify a stronger existing-card review policy, and what do current US large-bank aggregates add to the discussion?

**Recommendation.** Retain the interpretable scorecard and boosting challenger as research benchmarks. Investigate calibration and segment behavior before proposing a changed operating policy. The challenger has higher ROC AUC in the historical snapshot, while the scorecard has slightly higher average precision. The available evidence does not establish an improvement in assumed review cost or realized losses.

## Client-level evidence

The UCI study covers existing Taiwanese card customers in 2005. Both models receive identical training budgets and predictors. Duplicate feature vectors stay together across the four data roles. The known historical test has {test['rows']:,} clients, {test['defaults']:,} defaults and {prevalence:.2%} default prevalence.

- Scorecard ROC AUC: **{card['models']['scorecard']['auc']:.4f}**; challenger: **{card['models']['challenger']['auc']:.4f}**.
- Calibrated Brier: **{card['models']['scorecard']['brier']:.5f}** and **{card['models']['challenger']['brier']:.5f}**. Both are worse than their raw-score Brier values; the frozen calibration is preserved.
- Historical scenario cost: **{card['models']['scorecard']['cost_units']:,}** and **{card['models']['challenger']['cost_units']:,}** under 5 units per missed default and 1 per nondefault client flagged. The paired interval for the difference in cost per client spans **[{card['paired_uncertainty']['cost_per_client']['ci95'][0]:.4f}, {card['paired_uncertainty']['cost_per_client']['ci95'][1]:.4f}]**.
- At **10%** review capacity, the challenger flags **{int(q.flagged):,} clients**, capturing **{q.capture_rate:.1%}** of observed defaults; **{q.precision:.1%}** of the queue defaults. This is workload/concentration evidence, not a measured effect of intervention.

SQLite and separate metric calculations reconcile the supplied clients, frozen threshold decisions, calibration, ranking and capacity queues. **{len(audit['controls'])} controls pass with zero recorded violations.** Invalid identities, labels and probabilities fail before reports are produced.

## US market context

The separate Philadelphia Fed FR Y-14M pack covers {len(market):,} quarters through {latest['quarter']}. Latest reported balances are **USD {latest['balances_bn']:,.2f} billion**. The 90+ days past due rate is **{latest['dpd90_balances_pct']:.2f}% by balances** and **{latest['dpd90_accounts_pct']:.2f}% by accounts**. Annualized net charge-offs are **{latest['nco_annualized_pct']:.2f}%**. The market pack preserves the source's population, denominator and stock/flow definitions and supplies quarter-over-quarter/year-over-year checks.

These aggregates describe US large-bank card conditions. They cannot turn the Taiwanese cohort into a US borrower sample, identify a customer's default probability or establish a link from a market change to this fitted model.

Relative to the year-earlier quarter, 90+ balance delinquency changed by **{next(item['yoy_change_pp'] for item in market_report['year_on_year_indicators'] if item['metric'] == 'dpd90_balances_pct'):+.2f} percentage points**, while the below-660 share of new accounts changed by **{next(item['yoy_change_pp'] for item in market_report['year_on_year_indicators'] if item['metric'] == 'new_sub660_accounts_pct'):+.2f} points**. Portfolio mix and credit-line allocation merit separate investigation; the aggregate pattern does not identify a causal explanation.

## Conditions for a stronger future decision

Obtain an appropriate independent, time-indexed borrower sample with established information availability and mature outcomes. Assess calibration transport and segment errors, confirm the operational cost and review capacity, and measure the intervention's effect before making a business-benefit claim. Outcomes already inspected here cannot become an unseen holdout through repartitioning or relabeling.

The present deliverable is a reproducible research and reporting system with explicit limits. Evidence, source hashes, SQL queries and generated charts are linked in [the research report](RESEARCH_REPORT.md).
"""
    return research, memo


def build(root: Path, output: Path) -> list[Path]:
    card_dir, market_dir = root / "reference/card_clients/replay", root / "reference/us_market"
    audit_dir = root / "reference/card_clients/sql_review"
    sources = {"Card replay summary": card_dir / "summary.json", "Calibration bins": card_dir / "calibration_bins.csv", "Decision queues": card_dir / "decision_queue.csv", "Independent SQL review": audit_dir / "review.json", "US quarterly pack": market_dir / "quarterly.csv", "US market report": market_dir / "report.json"}
    for path in sources.values():
        if not path.is_file():
            raise ValueError("Required executed evidence is missing: " + str(path))
    card, audit, market_report = [load_json(sources[key]) for key in ("Card replay summary", "Independent SQL review", "US market report")]
    calibration, queues, market = [pd.read_csv(sources[key]) for key in ("Calibration bins", "Decision queues", "US quarterly pack")]
    validate_evidence(card, calibration, queues, audit)
    market = market_frame(market)
    validate_market(market, market_report)
    identities = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
    if audit.get("queue_assignment_reference_checked") is not True:
        raise ValueError("Independent individual queue membership reconciliation required")
    for name in ("summary.json", "holdout_predictions.csv", "partition_manifest.csv", "decision_queue.csv", "queue_assignments.csv"):
        path = card_dir / name
        if not path.is_file() or audit.get("source_sha256", {}).get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Independent review does not bind this card replay input: " + name)
    sql_queue = audit_dir / "decision_queues.csv"
    if not sql_queue.is_file() or audit.get("report_sha256", {}).get("decision_queues.csv") != hashlib.sha256(sql_queue.read_bytes()).hexdigest():
        raise ValueError("Independent SQL queue report identity mismatch")
    (output / "figures").mkdir(parents=True, exist_ok=True)
    theme()
    model_figure(card, output / "figures/model_comparison.png")
    calibration_figure(calibration, output / "figures/calibration.png")
    queue_figure(queues, card["partitions"]["test"]["default_rate"], output / "figures/review_capacity.png")
    market_figure(market, output / "figures/us_card_market.png")
    origination_figure(market, output / "figures/origination_mix.png")
    report, memo = build_markdown(card, calibration, queues, audit, market, market_report, identities)
    (output / "RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    (output / "DECISION_MEMO.md").write_text(memo, encoding="utf-8")
    return [Path("RESEARCH_REPORT.md"), Path("DECISION_MEMO.md"), *(Path("figures") / name for name in ("model_comparison.png", "calibration.png", "review_capacity.png", "us_card_market.png", "origination_mix.png"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Reject stale generated Markdown and figures")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            files = build(ROOT, out)
            changed = [str(path) for path in files if not (ROOT / "reports" / path).is_file() or (ROOT / "reports" / path).read_bytes() != (out / path).read_bytes()]
            if changed:
                raise SystemExit("Stale generated reports: " + ", ".join(changed))
        print("RESEARCH_PRESENTATION_CURRENT")
    else:
        files = build(ROOT, ROOT / "reports")
        print(f"RESEARCH_PRESENTATION_BUILT {len(files)} files")


if __name__ == "__main__":
    main()
