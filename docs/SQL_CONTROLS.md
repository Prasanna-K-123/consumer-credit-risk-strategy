# Independent SQL and metric controls

The `card_audit` package reviews the published calibrated customer scores. It
does not import the modeling implementation, fit a model, change a threshold,
or select a policy. SQLite queries and a separate Python calculation must agree
before it writes a successful review.

These are **postpublication diagnostics of an already inspected historical
holdout**. They do not create a new untouched test or establish prospective
performance. The data concern existing Taiwanese card customers in 2005;
review flags are not lending approvals or proof that an intervention works.

## Run

From the repository root, using Python 3.12 and its standard-library SQLite:

```bash
python -m card_audit \
  --predictions reference/card_clients/historical/holdout_predictions.csv \
  --partition-manifest reference/card_clients/historical/partition_manifest.csv \
  --summary reference/card_clients/historical/summary.json \
  --output artifacts/sql_review_historical
```

The same command accepts replay files. Add
`--queue-reference PATH/decision_queue.csv` to compare the model pipeline's
queue results with the independently computed SQL counts and rates. If
`--queue-assignments PATH/queue_assignments.csv` is supplied, every individual
customer/model/capacity flag must also match SQL's independent ordering; equal
aggregate counts cannot conceal different tied-customer selections. If
`--summary` is omitted, it defaults to `summary.json` beside the predictions.
Output always goes to the requested directory; source evidence is read only.

## Controls and their purpose

| Control | What is checked |
|---|---|
| Exact CSV contracts | Headers and required fields match the source interface. IDs and groups retain canonical integer representations. Outcomes must be exactly zero or one; probabilities must be finite and in `[0,1]` before any numerical clipping. |
| Source population | Each manifest/customer ID is unique. The prediction ID–group mapping equals the manifest's test population exactly, exposing missing, extra, wrong-role and mismatched-group rows. |
| Partition separation | All four study roles are present. A model-input group cannot cross roles. SQL independently checks this constraint and reconciles every role's row/group counts with the supplied research summary. |
| Threshold decisions | SQLite derives TP, FP, FN, TN, flag counts, Brier and log loss from the calibrated scores and each model's recorded policy-validation threshold. Confusion counts and historical cost results must agree with the summary. |
| Ranking metrics | SQL aggregates complete tied-score groups and uses window sums. Python separately counts positive/negative rank pairs and integrates precision over whole tied blocks. AUROC, average precision, Brier and log loss must agree. |
| Calibration and segments | Ten probability bands reconcile to the full test population and default count. Sex and age summaries support descriptive error inspection; they do not certify fairness. Empty probability bands are omitted, rather than assigned invented rates. |
| Fixed review capacities | SQL `ROW_NUMBER()` orders probability descending and source ID ascending, taking `floor(capacity × population)`. Capacities are 0%, 1%, 5%, 10%, 20%, 40% and 100%. Cutoff ties and exact workload/default counts are reported. |
| Failure behavior | Explicit exceptions remain active under `python -O`. Malformed data and inconsistent reference results cannot produce a pass message. |

Sex is used only in audit summaries, matching the source model's exclusions.
Age is a source-model predictor as well as a descriptive segment. Audit
segments overlap by design: a customer contributes to one sex group and one
age group, so adding every segment row would double-count the population.

## Metric definitions

AUROC counts a positive/negative pair as 1 when the positive score is higher,
0 when lower, and 0.5 for an exact score tie. Average precision adds
`positive_count_in_block × cumulative_precision` over descending distinct
score blocks, then divides by the number of positives. This prevents row order
within tied scores from changing the metric.

AUROC is undefined for a one-class sample. Average precision is undefined when
there are no positive outcomes; it equals 1 when every outcome is positive.
Undefined values remain JSON `null`, not zero or a fabricated perfect score.
Brier and log loss remain defined. Log loss clips validated probabilities at
Python's binary64 machine epsilon solely to make logarithms at 0 and 1 finite.

The review uses tolerance `1e-10` for SQL/Python metrics and `1e-7` against the
published research metrics, accommodating the source CSV's rounded probability
serialization and fitted-library numerical differences. Counts must agree
exactly. Optional queue-rate/cutoff comparisons use `1e-9`; their counts and
declared tie policy must agree exactly.

The original error-cost scenario is preserved: **5 units per missed default
plus 1 unit per flagged nondefault**. These units are not money, loss severity,
profit or measured intervention benefits. The zero-capacity and full-capacity
rows expose the unflagged-all and flagged-all endpoints. Capacities are fixed
diagnostics, not selected on the known test outcomes.

## Files produced

| File | Contents |
|---|---|
| `review.json` | Pass status, input/query/output-CSV SHA256 identities, exact controls, partition counts, independent metrics and declared tolerances. Optional supplied queue summaries/assignments receive their own input hashes. |
| `population.csv` | Row and unique input-group counts for every study role. |
| `confusion.csv` | Recorded-threshold decision accounting and probability losses for each model. |
| `calibration.csv` | Nonempty probability bands with observed/expected rates and counts. |
| `segments.csv` | Descriptive sex and age outcome/error summaries. |
| `decision_queues.csv` | Fixed-capacity workload, capture, precision, cutoff ties and scenario costs. |
| `card_review.sqlite` | The actual typed manifest, customer, policy, prediction and capacity tables used by the queries. |

Queries live in `card_audit/sql/` and can be inspected and rerun. The generated
database is a local analytical artifact, not a deployed database service. SQL
queries using `LN` and `CLIP` expect the functions registered by the Python
review; raw table queries work in a normal SQLite client.

## Interpreting the historical review

The verified historical files contain 30,000 manifest customers and 6,001 test
customers, including 1,295 defaults. The independent calculation reproduces
AUROC 0.781429 for the scorecard and 0.789673 for the challenger. Average
precision is 0.569715 and 0.565780 respectively: a higher AUROC does not imply
better performance under every ranking metric.

At the previously recorded thresholds 0.15 and 0.21, the scorecard flags 3,219
customers and catches 1,062 defaults; the challenger flags 2,050 and catches
873. The different workloads must remain visible when comparing their
scenario costs. This calculation corroborates the serialized scores and the
research summary; it does not independently re-extract the original Excel
data, establish original-vintage availability, validate a causal intervention,
or certify a production model. The source-to-score replay provides the
separate data/model reproduction route.

## Tests

```bash
python -m unittest discover -s tests -p 'test_sql*.py' -v
```

Analytic tied-score examples, one-class cases, exact population corruption,
malformed probabilities/outcomes, deliberately changed reference results,
capacity endpoints, exact tied-customer membership and optimized-Python failure behavior
exercise the controls. Repeated report generation is byte stable for the
scientific JSON/CSV outputs in the fixture. Database-file byte identity across
different SQLite versions is not claimed.
