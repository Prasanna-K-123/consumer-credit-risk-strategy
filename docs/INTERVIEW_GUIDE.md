# Explaining the project

Use the code and output to answer these questions in your own words. Reproduce the study before treating any result as part of your project experience.

## The problem and the populations

**What decision does the model support?** It ranks historical existing-card clients for a capacity-limited review queue. A flag is a research classification, not a lending approval or proof that an intervention helps.

**Why two datasets?** UCI supplies Taiwanese customer-level outcomes and payment histories for the modeling case. The Philadelphia Fed supplies genuinely US large-bank card aggregates for market reporting. Their units of observation and geographical coverage differ; they are not merged into a fictitious US training set.

**What is the label?** Next-month default in the UCI source snapshot. Six months of explanatory payment/bill information precede the labeled month. The study is a single historical cohort with customer-group partitions; it does not demonstrate chronological transport or original-vintage availability.

## The experiment

**Why group duplicate inputs?** An identical predictor vector in both training and evaluation could make evaluation overly optimistic. Groups are defined from model inputs and assigned wholly to one data role. The partition manifest makes the assignment auditable. Grouping does not eliminate every source of dependence or prove independent customer identities.

**Why four roles?** Training fits the base models; calibration fits a separate regularized logistic probability mapping; policy validation selects the stated decision threshold; test measures the frozen choices. Both models use the same population and predictor budget. Outputs and original choices remain preserved when a result is disappointing. The recorded calibration slopes are positive; the fitting algorithm does not impose a positive-slope constraint.

**How does the scorecard work?** Training-only quantile bins receive smoothed weight-of-evidence values, which enter logistic regression. Follow the binning and unknown-value paths in the implementation. Explain how the transform is learned without using calibration, policy or test outcomes.

**What does the challenger add?** Histogram gradient boosting can represent nonlinear response and interactions. More flexibility earns its place only through relevant evaluation; a small ranking gain alone does not establish lower loss or a superior operating policy.

## Metrics and decisions

**Why report ROC AUC and average precision?** ROC AUC measures pairwise ranking. Average precision describes the precision/recall tradeoff and should be compared with sample prevalence. Neither is a probability-calibration measure.

**Why Brier and log loss?** Both evaluate probability quality with different penalties. Calibration can worsen a later sample's Brier score, which happens in this preserved experiment. This does not authorize selecting a replacement method using the same test outcomes and calling the result confirmatory.

**What is the 5:1 cost?** An explicit sensitivity assumption: five units per missed default plus one unit per nondefault client flagged. It is not a measured dollar loss, loss-given-default estimate or profit. A top-risk queue answers a separate capacity question and uses source ID to resolve score ties deterministically.

**What does the bootstrap interval cover?** Paired resampling of model-input groups with fitted models, calibration and thresholds fixed. It describes conditional sampling variation in these outcomes, not all model-fitting/selection uncertainty or future regime changes. The model cost difference is uncertain even when a ranking difference appears stronger.

## CART segmentation and policy sensitivity

**What does the tree add?** The executed Python CART benchmark supplies short,
auditable decision paths and a comparison with a simpler segmentation model.
It uses Gini splits, maximum depth three and a minimum training-leaf setting of
200. The fitted tree has eight leaves; its smallest training leaf contains 211
clients. Tree construction uses training outcomes, calibration uses its separate
partition and threshold choices use policy-validation outcomes. Inspect
[`tree.txt`](../reference/card_strategy/tree.txt) and
[`leaf_segments.csv`](../reference/card_strategy/leaf_segments.csv) to connect
each rule to its support and observed outcome rate in each data role. These
are Python CART results, not SAS or CHAID execution.

**Did CART improve the model?** It has lower historical AUC (0.7445) than both
existing benchmarks. At the original 5:1 cost assumption, CART costs 3,433 units,
versus 3,322 for the scorecard and 3,287 for boosting. The added value is
explainable segmentation and sensitivity evidence, not a new performance-win
claim. At 1:1, its observed cost is 1,002 versus 1,025 for each other model;
different decision weights can change a comparison, and these retrospective
counts do not establish a reliably superior model.

**How are the sensitivity policies selected?** Five declared missed-default
costs (1, 2, 5, 10 and 20) are compared with a false-alarm cost of one. For each
model and ratio, the threshold minimizes policy-validation cost over
0.01, 0.02, ..., 0.99; equal minima choose the lowest threshold. The known
evaluation outcomes describe those policies rather than choosing a preferred
ratio. See [`policy_sensitivity.csv`](../reference/card_strategy/policy_sensitivity.csv).

**Why does CART flag everyone at ratios 10 and 20?** Under the stated error
cost, missing a default becomes expensive enough that policy validation selects
the lowest threshold. All 6,001 evaluation clients then exceed it, leaving
4,706 false alarms and no missed defaults. An all-review policy matches the
simple all-review baseline in this scenario. It might exceed a real team's
capacity; a cost assumption does not supply staff, prove an effective
intervention or establish financial profit. The separate fixed-capacity queues
answer the constrained-workload question.

**How do ties matter for a shallow tree?** Clients in one leaf share its raw
probability and calibrated score, so many ranks tie. Fixed-capacity queues break
ties by source ID, never by default outcome. Membership is reproducible, but
ordering within a tied leaf does not demonstrate extra model discrimination.

**Is this a new independent validation?** The extension uses the already known
Taiwan cohort and its original partitions. Separating fit, calibration and
policy roles still prevents specific leakage, but does not make previously
inspected evaluation outcomes unseen. The tree and sensitivity comparisons are
postpublication descriptive work. They do not extend the original paired
bootstrap to CART or establish transport to future US borrowers.

## Independent software checks

**What makes the SQL review useful?** Inspect the executable SQL population, confusion, calibration, segment and queue queries. Explain the join keys, uniqueness checks, window-function ordering and exact count reconciliation. A second metric implementation uses score groups to handle ranking ties and checks direct probability arithmetic.

**What was strengthened in the historical replay?** Input contracts reject invalid probabilities before clipping, fractional labels, mismatched shapes and malformed identities. Integrity failures use explicit exceptions that remain effective under optimized Python. Protocol, source and result identities are checked rather than trusting a success label.

**How do you reproduce the results?** Start with the root README commands, acquire the exact official source bytes, run the scientific replay, execute the SQL review, build the US market pack and generate the reports. Distinguish a successful current execution from historical receipts. Changed source bytes should stop reproduction and require a reviewed source update.

## Market reporting and judgment

**Why can account- and balance-based delinquency disagree?** They use different denominators. A small share of delinquent accounts can carry a larger share of balances. Compare rates only after checking the source's population and delinquency definition.

**What is a stock versus a flow?** Balances and commitments are quarter-end stocks; new accounts, initial credit commitments and purchases are quarterly flows. Net charge-offs are a quarterly flow rate annualized by the source. A ratio must respect those definitions.

**What does the market history prove?** It reproduces a particular release's retrieved, potentially revised series. It does not reconstruct every publication vintage or identify causal effects. Missing values remain missing, and source rounding limits precision of inferred ratios.

**What would be required for a bank decision?** Appropriate independent borrower histories, known feature availability, mature outcomes, validation across time and relevant segments, grounded cost/capacity constraints, model governance, security and measured intervention effects. A research repository demonstrates methods and judgment without implying those approvals already exist.

## A ninety-second explanation

Practice this structure in your own words after completing the walkthrough:

> The project asks how to compare credit-risk models and allocate limited review
> capacity. The client study uses 30,000 existing Taiwanese cardholders from
> 2005. I would describe its results as historical research, rather than US
> lending performance.
>
> The comparison uses a transparent WoE/logistic scorecard and a boosting
> challenger. Duplicate model-input groups stay together, and training,
> calibration, policy selection and evaluation have separate roles. The
> challenger has higher AUC, but the scorecard has higher average precision.
> Calibration worsens both models' test Brier scores, and the paired interval
> does not establish an improvement in the assumed decision cost.
>
> At 10% review capacity, the challenger's 600-client queue contains 33.4% of
> observed defaults, with a 72% queue default rate. That describes concentration
> and workload; it does not show that reviewing those clients prevents default.
>
> A retrospective shallow CART benchmark makes segments inspectable. Cost-ratio
> sensitivity shows how assumed missed-default costs can push review workload
> up to the entire population.
>
> Independent SQL reconciles the population, decisions and queue membership.
> A separate official US bankcard market pack covers 56 quarters, with its own
> denominators and source checks. Before a bank could use the client model, I
> would require appropriate new borrower histories, temporal validation,
> grounded costs and evidence about the intervention.

Use the explanation as a sequence of ideas, rather than memorizing a script.
If a detail is forgotten, locate its evidence and explain how it is calculated.

## Practical walkthrough

The exercises below provide observable evidence of understanding. A completed
repository or a successful automated test suite does not complete them for an
applicant. Use a fresh output directory for every reproduction command.

| Exercise | Evidence to demonstrate |
|---|---|
| Reproduce the complete study | Run the root README commands; locate the new execution receipt, scientific replay, SQL evidence and market outputs. Explain what each stage actually executed. |
| Trace one input row | Follow a source ID through the partition manifest and prediction output; explain the distinction between identity and model features. |
| Explain leakage control | Locate `feature_groups` and `partition` in `consumer_cards/data.py`, then trace training-only WoE fitting in `consumer_cards/models.py`. Show why test outcomes cannot enter those fits. |
| Recalculate a decision | Choose one model's frozen threshold, compute its confusion counts from predictions, and reconstruct `5 × missed defaults + nondefault clients flagged`. Explain the units. |
| Challenge a headline | Explain why AUC 0.7897 versus 0.7814 does not settle policy selection. Locate average precision, raw/calibrated Brier and the paired cost interval. |
| Inspect a queue | Reproduce the SQL ranking below; explain `ROW_NUMBER`, descending probability, deterministic ties and the floor rule. Reconcile the 600-client queue. |
| Walk one CART path | Follow a client through `reference/card_strategy/tree.txt`; reconcile its leaf in the predictions and leaf-segment outputs. Explain the training support and the limits of the segment's outcome rate. |
| Reconstruct cost sensitivity | Read one row of `policy_sensitivity.csv`; reconstruct the cost from missed defaults and false alarms, compare with review-all/review-none and explain which partition chose the threshold. |
| Explain a constrained decision | Contrast CART's all-review result at 10:1 with a 10% capacity queue; explain the tied-score rule and the missing evidence needed for profit or treatment-effect claims. |
| Demonstrate failure handling | In temporary copies, introduce a duplicate source ID or invalid probability and show a validation failure. Preserve committed reference evidence. |
| Read a market indicator | Trace a latest-quarter value back to its official source field, unit and denominator. Explain a stock/flow distinction and why missing is different from zero. |
| Explain transport limits | State exactly what data are needed for US borrower-level, acquisition, temporal and intervention claims. |
| Defend the bank-model conclusion | Use the separate bank-validation case audit to explain how pooled AUC 0.916 coexists with 9 of 10 later failures missed. Identify the subgroup as post hoc. |
| Defend the customer-policy conclusion | Explain random assignment, development/evaluation separation, the comparator and the uncertainty interval. Identify assumed margins and contact costs. |

After running the SQL component, open its `card_review.sqlite` database with
Python's built-in `sqlite3` module or a SQLite client. This query is intentionally
self-contained so its ranking and counts can be explained without a helper:

```sql
WITH ranked AS (
  SELECT p.model, p.source_id, c.outcome,
    ROW_NUMBER() OVER (
      PARTITION BY p.model
      ORDER BY p.probability DESC, p.source_id ASC
    ) AS position,
    COUNT(*) OVER (PARTITION BY p.model) AS population
  FROM predictions p
  JOIN customers c USING (source_id)
)
SELECT model,
  COUNT(*) AS reviewed,
  SUM(outcome) AS defaults_reviewed,
  SUM(outcome) * 1.0 / COUNT(*) AS queue_default_rate
FROM ranked
WHERE position <= CAST(0.10 * population AS INTEGER)
GROUP BY model;
```

To obtain default capture, divide defaults reviewed by all 1,295 defaults in
the evaluation population. Explain why queue default rate and default capture
answer different questions. Explain why neither measures prevented defaults.

## Readiness rubric

Assess the explanation through actual demonstrations, with notes about what
worked and what still needs practice. No readiness result is asserted here.

| Area | Evidence of readiness | Evidence that more practice is needed |
|---|---|---|
| Problem and scope | States population, target, decision and transfer limits accurately | Calls the Taiwan sample US lending data or the queue an approved lending policy |
| Statistical design | Traces each data role and explains leakage, metrics and conditional uncertainty | Tunes on known evaluation outcomes or treats AUC as proof of savings |
| Code and SQL | Navigates the implementation, reproduces a count and explains joins/ties | Can repeat reported numbers but cannot reconstruct them |
| Business judgment | Connects workload and uncertainty to a restrained recommendation | Converts assumed costs into measured profit or mistakes concentration for treatment effect |
| Communication | Gives a concise explanation, then supports follow-up questions with evidence | Relies on unexplained terminology or conceals a failed comparison |

A useful stopping point is being able to complete every walkthrough task,
explain each key decision without reading prepared answers, and handle a new
follow-up about an assumption. Record uncertainty honestly and investigate it;
the aim is sound reasoning, not a claim to know every possible fact.
