# Consumer credit risk: methods and interpretation

This project separates a historical credit-card model comparison from a new
software workflow for acquisition, replay, verification, scoring and capacity
scenarios. The historical test results have already been published. Replaying
them, hardening the implementation or adding an analysis does not create a new
untouched holdout.

## Data and prediction question

The source is I-Cheng Yeh's **Default of Credit Card Clients** dataset,
[UCI DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H), distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It contains 30,000
Taiwan credit-card clients from 2005. The binary target is default payment in
the next month. The observations describe existing clients, rather than a
representative sample of new applicants.

The historical model comparison uses the same 20 predictors for both models:

| Predictor group | Fields |
| --- | --- |
| Credit limit | `LIMIT_BAL` |
| Age | `AGE` |
| Repayment status | `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` |
| Bill amounts | `BILL_AMT1` through `BILL_AMT6` |
| Payment amounts | `PAY_AMT1` through `PAY_AMT6` |

`ID`, `SEX`, `EDUCATION` and `MARRIAGE` are excluded from fitting. Original
source identifiers support traceability; sex is retained only for descriptive
audit tables. Age remains a model input. Excluding selected columns does not
establish fairness or eliminate proxy effects.

The expected XLS SHA-256 is
`30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.
Acquisition must check this identity before parsing or fitting. A changed
upstream file requires investigation; it must not be accepted by silently
replacing the reference hash.

## Historical comparison design

Exact duplicate vectors of the 20 model inputs form groups. Every row in a
group stays in one partition, including groups containing different labels.
This prevents identical model inputs appearing on both sides of the split.
The nominal group allocation is 60% training, 10% calibration, 10% policy
validation and 20% test; actual row proportions differ slightly.

| Partition | Rows | Distinct model-input groups | Defaults |
| --- | ---: | ---: | ---: |
| Training | 18,007 | 17,861 | 3,964 |
| Calibration | 2,992 | 2,977 | 680 |
| Policy validation | 3,000 | 2,977 | 697 |
| Test | 6,001 | 5,954 | 1,295 |

The historical split seed is `20261010`. There are 29,769 distinct input
groups and 231 additional duplicate-input rows. This is a random group
holdout within one cohort, not out-of-time validation. Monthly predictor
ordering does not establish performance across later cohorts or countries.

The transparent reference is training-only weight-of-evidence transformation
with five quantile bins for continuous predictors, followed by logistic
regression with `C=1`. The challenger is histogram gradient boosting with
200 iterations, at most 31 leaves, learning rate 0.05, L2 regularization 10
and early stopping disabled. Both use identical predictors and partitions;
their different preprocessing is part of the declared comparison.

Each fitted model receives a separate regularized logistic calibration on
the calibration partition, using clipped log odds of its raw probabilities.
Valid endpoints are clipped only for numerical evaluation of log odds;
nonfinite or out-of-range probabilities must be rejected before clipping.

Each calibrated model's threshold is selected on the policy-validation
partition from `0.01, 0.02, ..., 0.99`, minimizing this illustrative loss:

\[
L(y,p;t)=5\,\mathbf{1}(y=1,p<t)
          +\mathbf{1}(y=0,p\geq t).
\]

The weights are error-cost units, not money. They are not estimates of
revenue, loss given default, exposure at default or lending profit. Historical
thresholds are 0.15 for the scorecard and 0.21 for the challenger. The test
partition is used to report the already selected models and thresholds.

## What the historical results support

| Test measure | Scorecard | Challenger |
| --- | ---: | ---: |
| ROC AUC | 0.781429 | 0.789673 |
| Calibrated Brier score | 0.130794 | 0.130042 |
| Uncalibrated Brier score | 0.130485 | 0.129523 |
| Illustrative total error cost | 3,322 | 3,287 |

The challenger has a higher measured AUC and lower measured Brier score and
cost. The 2,000-draw paired group bootstrap gives a challenger-minus-scorecard
AUC interval of `[0.000555, 0.016102]`. The Brier difference interval is
`[-0.002191, 0.000646]`; the cost-per-client difference interval is
`[-0.032834, 0.019433]`. Both latter intervals cross zero. Therefore the
measured cost reduction does not establish a reliable improvement under the
declared comparison. Calibration also **worsens test Brier score for both
models** relative to their raw probabilities. It must not be advertised as an
observed improvement merely because a calibrator was fitted.

Each bootstrap draw samples test input groups with replacement and includes
all rows belonging to a sampled group. The intervals condition on the fitted
models, calibrators and thresholds. They omit training/refitting uncertainty,
source-selection uncertainty and future population drift. They are not
universal performance guarantees.

Sex and age-group summaries describe the historical sample. They can reveal
differences worth investigating, but they do not certify fairness or establish
discrimination, regulatory compliance or causal effects.

## New software and capacity scenario

The package provides separate entry points rather than silently mixing data
acquisition, research fitting and operational scenarios:

| Command | Purpose |
| --- | --- |
| `python -m consumer_cards acquire` | Obtain the original source and check the expected XLS hash. |
| `python -m consumer_cards replay` | Refit the fixed historical design and emit local evidence. |
| `python -m consumer_cards verify` | Check local replay evidence against the frozen historical reference. |
| `python -m consumer_cards score` | Refit the original models from the checked source, then score a validated input batch. |
| `python -m consumer_cards queue` | Rank a scored batch for a declared capacity scenario. |

Use each command's `--help` for its input, output and artifact arguments. These
are procedures, not assertions that a run has passed. Execution results belong
in a receipt with the actual environment, input identities and checks run.

Fitting, calibration and prediction temporarily limit native BLAS/OpenMP
pools to two threads through `threadpoolctl`. Calls restore the previous
thread limits, including after errors, and do not change environment variables.
The limit bounds numerical-worker overhead on a shared machine; it does not
make the package a concurrent request-serving system. The model-version
fingerprint includes source, protocol, implementation and dependency versions.

The queue orders calibrated default probabilities from highest to lowest,
breaking ties by ascending `source_id`. For a batch of `n` rows and capacity
fraction `c`, it selects exactly `floor(c * n)` rows. Source identifiers must
be unique, scores finite and within `[0,1]`, and capacity finite and within
`[0,1]`. Ranking determines a review order; it does not itself make an approval,
denial, collection or customer-contact decision.

The retrospective queue is a simultaneous-batch scenario: the complete batch
is available before allocation. It is not a simulation of sequential arrivals
or a learned deployment policy. When outcomes from the historical test batch
are summarized at different capacities, that is a **postpublication
descriptive extension**. Looking at those outcomes must not be presented as
fresh validation, causal treatment benefit or an opportunity to retune the
historical models while preserving an untouched-test claim.

## Verification and interpretation boundaries

Original bytes are preserved under `reference/card_clients/historical`.
Local replay outputs must be separate. Integrity checks must remain effective
under optimized Python execution, reject invalid labels and shapes before
coercion, and verify more than headline metrics. Relevant evidence includes
source/protocol identity, partitions, predictions, calibration parameters and
bins, subgroup summaries, paired uncertainty and the corresponding artifact
identities. Exact data and count checks and any numerical comparison
tolerances must be explicit. A verifier must never rewrite a historical
reference to make a failure disappear.

This project does not establish US consumer-card performance, out-of-time
generalization, a profitable acquisition policy, a production deployment,
regulatory adequacy or fairness certification. Its defensible contribution is
reviewable consumer-credit software, a reproducible historical comparison and
clear separation of measured evidence from scenario assumptions. See
[history and provenance](CARD_HISTORY.md) for the limits of the original
protocol record.

## Retrospective CART and cost sensitivity

The separately recorded `card_strategy/PROTOCOL.json` fixes a depth-three Gini
CART with a minimum of 200 training rows per leaf and seed 20261030. It uses the
same 20 predictors and original data roles, a separate logistic calibrator and
the same 0.01–0.99 policy-threshold grid. Five missed-default cost ratios
(1, 2, 5, 10, 20), each against false-alarm cost 1, are all reported alongside
review-all/review-none benchmarks. Exact threshold ties choose the lowest value.
Three capacity scenarios use probability descending/source-ID ascending ordering.

The tree is fitted only on training labels; calibration and threshold selection
use their separate roles. Paths and support/default counts for every leaf and
all four roles are exported. PAY status codes are split numerically. These
controls do not erase prior knowledge of the historical evaluation outcomes.
No new untouched experiment, SAS/CHAID execution, comparative significance or
monetary operating benefit is claimed. The original scorecard/boosting protocol
and its outputs remain unchanged.

See the [generated strategy report](../reports/STRATEGY_REPORT.md) and
[scope/access notes](COVERAGE_AND_ACCESS.md).
