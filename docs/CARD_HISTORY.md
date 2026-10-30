# Consumer credit risk: history and provenance

This standalone project carries forward an existing UCI credit-card model
comparison and adds a software interface for acquisition, replay,
verification, scoring and a capacity-constrained review queue. It does not
present the historical results as a newly conducted untouched experiment.

## Source attribution

**Dataset:** I-Cheng Yeh, *Default of Credit Card Clients*, UCI Machine
Learning Repository,
[DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The new software uses the original Taiwan 2005 cohort. A consumer-card
workflow can be relevant to risk-analysis skills without making this sample
representative of current US cardholders or applicants. Dataset attribution
and terms apply separately from the software's license.

## Historical record

| Item | Recorded identity |
| --- | --- |
| Original XLS SHA-256 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| Historical protocol SHA-256 | `7376e59998b3563fbddca17c4ab1cd9dde1e089118a62dcba42940aa126f1ce3` |
| Timestamp inside the protocol | `2026-10-10T02:59:08.755108+00:00` |
| First inspected historical Git commit | `f4bb37d2d51625f2fc76c38504072211850c7f83` |
| Timestamp of that commit | `2026-10-10T03:34:55Z` |

The first inspected commit contains the source, protocol and results
together. The protocol's own timestamp is earlier, but a timestamp written
inside a file does not independently establish when its contents were fixed
relative to model fitting or result inspection. Consequently, this project
does **not** claim independently established preregistration. Hashes identify
the preserved bytes; they do not, by themselves, prove prospective research
chronology or external review.

The historical implementation separates training, calibration,
policy-validation and test partitions. That separation is inspectable in the
code and artifacts. It should be described as the recorded design rather
than used to claim a stronger independently witnessed chronology.

Original reference bytes are frozen under
`reference/card_clients/historical`. They are retained as historical evidence,
not silently repaired to match a later software revision. A new replay uses
separate outputs and reports any discrepancy against the historical record.

## Findings retained from the original comparison

The historical scorecard and boosting challenger have test AUCs of
**0.781429** and **0.789673**, calibrated Brier scores of **0.130794** and
**0.130042**, and illustrative total error costs of **3,322** and **3,287**.
The paired cost-per-client interval crosses zero. Both calibrators worsen
test Brier score relative to the respective raw model probabilities. These
results and failures are part of the evidence; the software must not turn
them into claims of demonstrated profit or uniformly improved calibration.

The [methodology](CARD_METHODOLOGY.md) describes the fixed model budgets,
group-separated partitions, conditional uncertainty and interpretation
boundaries.

## What changes in this standalone project

The new contribution is a focused consumer-credit package with explicit
commands for source acquisition, historical replay, evidence verification,
batch scoring and capacity allocation. Input-contract checks, audit coverage
and new tests can improve the software without changing the historical
scientific reference. Any changed research method, model specification or
target population belongs in a separately identified study.

The review queue ranks calibrated probabilities descending and uses ascending
`source_id` to resolve ties. It selects `floor(capacity * batch_size)` rows
from a complete batch. Retrospective outcome summaries for that queue are
postpublication descriptive scenarios. They are not new untouched-holdout
results, evidence of causal intervention benefit or an arrival-by-arrival
deployment simulation.

Replay verifies whether a fixed design can be reconstructed. Verification
receipts must identify what was actually checked and preserve failures and
limitations. Documentation of a command, a test file or an expected hash is
not proof that the command has executed successfully. No successful current
run, independent external replication, production use, US generalization,
regulatory approval or fairness certification is asserted by this history.
