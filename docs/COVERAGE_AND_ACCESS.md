# Scope, evidence coverage and access dependencies

Reviewed 30 October 2026. The project demonstrates consumer-credit research,
independent model challenge, SQL reconciliation, readable CART segments,
policy-cost sensitivity and official US market reporting. Each claim is tied to
executed outputs; the populations and decision assumptions remain explicit.

## US borrower evidence

The Taiwan 2005 card cohort is a historical existing-customer modeling example.
Its four data roles prevent direct training/calibration/policy reuse, but their
outcomes were already public when the strategy extension was designed. It is not
a fresh experiment, US underwriting validation or temporal transport test.

The US reporting component uses Philadelphia Fed aggregates. The Fed's
[FR Y-14M FAQ](https://www.philadelphiafed.org/surveys-and-data/y14-frequently-asked-questions)
restricts account microdata to the Federal Reserve System because the underlying
records contain confidential supervisory information. The New York Fed's
[Consumer Credit Panel FAQ](https://www.newyorkfed.org/microeconomics/faq) similarly
limits its microdata to Federal Reserve researchers and coauthors under its
data-provider contract. The public aggregates cannot be disaggregated into
authentic borrower histories.

No credential-free, appropriately licensed US card-client panel with documented
feature timing and mature outcomes was verified in this review. This describes
the sources checked, rather than a claim that no suitable dataset exists.
The [American Express competition](https://www.kaggle.com/competitions/amex-default-prediction)
is a possible separately scoped card dataset, requiring account/rules and source
audits. Its description does not establish representative US geography. A loan
dataset with a current status and issue month alone cannot establish a common,
mature future-default horizon.

A stronger borrower validation extension would need legitimate source access,
documented origination/statement dates, predictors available at scoring time,
mature outcomes, selection information and permission for the intended use.
Geography, population and operational target would be established before fitting.

## SAS and related methods

The shallow CART classifier is executed in Python and its exact numeric paths,
training support and role-specific outcomes are exported. It is not CHAID or an
executed SAS implementation. There is no demonstrated SAS runtime in the current
execution environment.

[SAS OnDemand for Academics](https://welcome.oda.sas.com/) provides a genuine SAS
environment. The official [registration FAQ](https://support.sas.com/ondemand/caq_new.html)
requires a SAS Profile followed by separate OnDemand registration. A user-controlled
registered account and usable runtime are needed before a SAS supplement can be
executed and its logs reconciled. Unexecuted code would not demonstrate SAS skill.

## Lending economics and regulatory scope

Cost ratios 1, 2, 5, 10 and 20 are transparent missed-default error weights, with
one unit per nondefault client flagged. They are not revenue, exposure at default,
loss given default, recoveries, lending profit or measured treatment effects.
The queue study measures concentration and workload; an intervention could have
no benefit even when the queue contains many future defaults.

CECL/CCAR models and regulatory approval are outside the implemented evidence.
Adding their names does not establish lifecycle loss forecasting or regulatory
adequacy. Those questions require appropriately defined exposures, cash flows,
loss outcomes, scenarios, estimation choices and validation evidence.

## Reproduction and explanation

The software can establish what was executed and checked. Applicant understanding
requires demonstrating the code, calculations and decision reasoning. The
[interview guide](INTERVIEW_GUIDE.md) provides practical exercises and explicit
readiness criteria; it does not record a completed applicant assessment.
