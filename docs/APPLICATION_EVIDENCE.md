# Project evidence for a credit-risk application

These descriptions summarize executed research. Use them as personal project
experience only after reproducing the work and explaining the implementation in
your own words. Preserve the distinction between historical observations,
assumed decision costs and measured business effects. Development dates and
responsibilities must match the work actually undertaken.

## Concise application-form answer

Consumer Credit Risk & Review Strategy — Python, SQL, scorecards, CART, policy sensitivity; Bank-Risk Model Validation — temporal evaluation, calibration, monitoring; Causal Customer Decisions — randomized experiments, targeting-policy evaluation.

The form asks for titles or keywords. Keep the detailed results for the CV,
linked evidence and interview. The three entries address consumer credit,
independent model challenge and customer-strategy testing respectively.

## Evidence-backed project bullets

### Consumer Credit Risk & Review Strategy

[Source and reproduction](https://github.com/Prasanna-K-123/consumer-credit-risk-strategy)
· [Decision memo](../reports/DECISION_MEMO.md)
· [CART and policy analysis](../reports/STRATEGY_REPORT.md)

- Compared a WoE/logistic scorecard and boosting challenger on 30,000 historical
  card customers, separating training, calibration, policy selection and a
  6,001-client evaluation; assessed ranking, probability quality and paired
  uncertainty.
- Evaluated review workload: a 600-client queue captured 33.4% of observed
  defaults at 10% capacity; independently reconciled model decisions and queue
  membership with SQL. This is historical concentration, not prevented losses.
- Added an interpretable depth-three CART benchmark and cost-ratio sensitivity:
  traced eight training-derived segments and showed how assumed missed-default
  costs change review workload, including an all-review policy at high ratios.
- Produced a separate 56-quarter US bankcard reporting pack from official data,
  covering delinquency, charge-offs and origination mix with source and
  reconciliation controls.

For a compact CV, select two bullets matching the target role and mention other
components in the methods line. The borrower sample is Taiwanese, from 2005; the US aggregates
are a different population and must remain identified separately. The 84,014
checked queue assignments are repeated model/capacity/customer combinations,
not 84,014 different customers.

The CART and cost-ratio extension is retrospective analysis of the same known
cohort. CART's AUC is 0.7445, and its 5:1 assumed cost is 3,433 units, compared
with 3,322 for the scorecard and 3,287 for boosting. Its purpose is transparent
segmentation and decision sensitivity, rather than a claim that another
algorithm improved performance. The executed implementation is Python CART;
it does not demonstrate SAS or CHAID execution.

### Bank-Risk Model Validation

[Source and reproduction](https://github.com/Prasanna-K-123/credit-risk-model-validation)
· [Decision memo](https://github.com/Prasanna-K-123/credit-risk-model-validation/blob/main/reports/DECISION_MEMO.md)

- Evaluated logistic and boosting bank-failure models on 237,253 FDIC bank-year
  observations using purged temporal roles, separate calibration and a frozen
  monitoring policy.
- Challenged pooled AUC of 0.916 with cohort-level evidence: the fixed policy
  missed 9 of 10 failures in a later historical subgroup, supporting restricted
  research use; independently checked annual counts and every failed-bank case.

The later subgroup is post hoc. Revised historical data and assumed availability
limit point-in-time claims. This is bank-failure monitoring, not a consumer
default model, regulatory approval or live risk-management deployment.

### Causal Customer Decisions

[Source and reproduction](https://github.com/Prasanna-K-123/causal-customer-decisions)
· [Decision memo](https://github.com/Prasanna-K-123/causal-customer-decisions/blob/main/reports/DECISION_MEMO.md)

- Evaluated campaign-targeting policies in a 64,000-customer randomized study,
  separating policy development from a 12,800-customer evaluation and comparing
  personalized targeting with simple alternatives.
- Used paired uncertainty and an independent weighting estimate to challenge
  the learned policy: its estimated incremental scenario value was
  -$0.049/customer, with a 95% interval of [-$0.193, $0.095], leaving superiority
  unestablished.

Value assumes a 40% margin and $0.05 contact cost; actual operating costs were
not supplied. The separate Criteo analysis concerns a released sample and does
not establish original-population causal profit.

## How to use the limited application space

Lead with consumer-card risk, then bank-model validation, then causal customer
decisions. Give each a decision question, a specific method and an interpretable
finding. The broader portfolio supports follow-up discussion; its eight studies
need not receive equal space on a role-specific CV.

Link the [credit-focused portfolio](https://prasanna-k-123.github.io/financial-decision-lab/portfolio.html#consumer-credit)
and the individual repositories. Keep project titles and population descriptions
consistent across the application, CV, GitHub and LinkedIn. Avoid unsupported
claims of improved bank profit, reduced realized losses, production adoption,
regulatory compliance or superior performance on future US borrowers.

## Before an interview

Complete the practical walkthrough in [the interview guide](INTERVIEW_GUIDE.md).
Be prepared to explain why the card challenger is not automatically the better
operating policy, why high assumed missed-default costs can produce an
all-review CART policy, why strong bank-model AUC coexists with missed failures, and
why a randomized customer experiment can still leave personalized targeting
unproven. These conclusions are the substantive decision stories behind the
implementation.

This document prepares project wording; it does not edit or certify a CV, submit
an application, or establish an applicant's interview readiness.
