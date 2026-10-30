# Decision memo: consumer-card risk evidence

**Question.** Does a flexible risk model justify a stronger existing-card review policy, and what do current US large-bank aggregates add to the discussion?

**Recommendation.** Retain the interpretable scorecard and boosting challenger as research benchmarks. Investigate calibration and segment behavior before proposing a changed operating policy. The challenger has higher ROC AUC in the historical snapshot, while the scorecard has slightly higher average precision. The available evidence does not establish an improvement in assumed review cost or realized losses.

## Client-level evidence

The UCI study covers existing Taiwanese card customers in 2005. Both models receive identical training budgets and predictors. Duplicate feature vectors stay together across the four data roles. The known historical test has 6,001 clients, 1,295 defaults and 21.58% default prevalence.

- Scorecard ROC AUC: **0.7814**; challenger: **0.7897**.
- Calibrated Brier: **0.13079** and **0.13004**. Both are worse than their raw-score Brier values; the frozen calibration is preserved.
- Historical scenario cost: **3,322** and **3,287** under 5 units per missed default and 1 per nondefault client flagged. The paired interval for the difference in cost per client spans **[-0.0328, 0.0194]**.
- At **10%** review capacity, the challenger flags **600 clients**, capturing **33.4%** of observed defaults; **72.0%** of the queue defaults. This is workload/concentration evidence, not a measured effect of intervention.

SQLite and separate metric calculations reconcile the supplied clients, frozen threshold decisions, calibration, ranking and capacity queues. **6 controls pass with zero recorded violations.** Invalid identities, labels and probabilities fail before reports are produced.

## US market context

The separate Philadelphia Fed FR Y-14M pack covers 56 quarters through 2026Q2. Latest reported balances are **USD 978.09 billion**. The 90+ days past due rate is **1.49% by balances** and **0.70% by accounts**. Annualized net charge-offs are **5.01%**. The market pack preserves the source's population, denominator and stock/flow definitions and supplies quarter-over-quarter/year-over-year checks.

These aggregates describe US large-bank card conditions. They cannot turn the Taiwanese cohort into a US borrower sample, identify a customer's default probability or establish a link from a market change to this fitted model.

Relative to the year-earlier quarter, 90+ balance delinquency changed by **-0.10 percentage points**, while the below-660 share of new accounts changed by **+2.44 points**. Portfolio mix and credit-line allocation merit separate investigation; the aggregate pattern does not identify a causal explanation.

## Conditions for a stronger future decision

Obtain an appropriate independent, time-indexed borrower sample with established information availability and mature outcomes. Assess calibration transport and segment errors, confirm the operational cost and review capacity, and measure the intervention's effect before making a business-benefit claim. Outcomes already inspected here cannot become an unseen holdout through repartitioning or relabeling.

The present deliverable is a reproducible research and reporting system with explicit limits. Evidence, source hashes, SQL queries and generated charts are linked in [the research report](RESEARCH_REPORT.md).
