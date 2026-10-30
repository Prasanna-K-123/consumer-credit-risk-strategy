# US consumer bankcard market risk pack

This component turns published US bankcard aggregates into a reproducible risk-reporting pack. It complements borrower-level modeling by showing portfolio composition, delinquency, payment behavior and new-account credit-line choices. It does not estimate an individual borrower's probability of default, identify a policy's causal effect, or reproduce a bank's confidential performance.

## Source and scope

The source is the Federal Reserve Bank of Philadelphia's [Large Bank Credit Card and Mortgage Data](https://www.philadelphiafed.org/surveys-and-data/large-bank-credit-card-and-mortgage-data). The pinned release is **2026Q2**, published **7 October 2026** on the source page. Both credit-card CSVs contain **56 quarterly observations, 2012Q3–2026Q2**. Their 35 balance metrics and 15 origination metrics are parsed using an explicit 50-metric dictionary.

The underlying source is FR Y-14M regulatory reporting. The public series cover large-bank consumer bankcards; they exclude small-business, corporate and consumer charge cards. Most balance statistics use open accounts that have not been charged off. Active-account variables instead use accounts with debit, credit or balance activity in the preceding 12 months. These populations must not be conflated. The [source FAQ](https://www.philadelphiafed.org/surveys-and-data/y14-frequently-asked-questions) describes the public aggregates and restricted account microdata.

These are nominal, non-seasonally-adjusted aggregates. No claim is made about all US consumers, a particular issuer, or an individual customer's behavior.

## Acquisition and integrity contract

Run from a checkout:

```bash
python -m us_card_market --output reference/us_market
python -m unittest discover -s tests -p 'test_market*.py' -v
```

No third-party Python package is needed. The canonical packaged manifest is [us_card_market/data/source_manifest.json](../us_card_market/data/source_manifest.json); it records official URLs, exact SHA-256 hashes, sizes, source pages and expected coverage. The downloader caches original bytes and actual UTC retrieval metadata locally under `.cache/us_card_market/`. It never substitutes synthetic or sample data for an unavailable official source. A changed download fails explicitly and requires source review before a manifest change.

The parser checks the entire CSV: exact ordered headers, strict numeric formatting, ordered unique quarters, complete pinned coverage, and the exact empty/source/website/email footer. An unexpected row is an error, including a misspelled quarter or an altered footer. A dollar value without its expected currency format, a malformed thousands separator, a percentage missing `%`, and an unexpected missing value are rejected.

There is one source null: **2012Q3 credit-line-decrease share**. It remains SQL `NULL` and a blank CSV cell. It is not a zero. The definitions page describes a 2012Q4 start for credit-line-change reporting, while the actual file supplies a 2012Q3 increase share. The pack preserves the supplied increase value and documents this distinction rather than deleting data or creating a replacement.

## Units and measures

Percentages are stored as **0-to-100 values**. A movement from 1.49 to 1.72 is +0.23 percentage points; relative percentage growth is a different calculation. Fields ending `_bn` are USD billions, `_mn` are million accounts, `_usd` are USD, and `_pct` are percentages. The generated dictionary includes each field's source header, population, unit and aggregation.

| Measure | Calculation and interpretation |
|---|---|
| Average balance per reported account | `1,000 × total balances (bn) / account count (mn)`; an average, not an active-account median |
| Aggregate balance-to-commitment ratio | `100 × total balances / total commitments`; line-weighted aggregate measure, not the median utilization of active accounts |
| New-account average initial limit | `1,000 × new commitments (bn) / new accounts (mn)`; originations represent assigned initial limits, not dollars borrowed |
| Below-660 implied average initial limit | Overall new-account average multiplied by below-660 commitment share divided by below-660 account share |
| At-least-660 implied average initial limit | Same calculation using the complementary account and commitment shares |
| Payment-category residual | `100 − minimum − partial − full payment shares`; an unclassified source residual, not a nonpayer estimate |
| Year-on-year rate change | Current percentage minus the percentage four quarters earlier, reported in percentage points |
| Year-on-year balance/account growth | Current level divided by the level four quarters earlier, minus one, multiplied by 100 |

Implied averages use rounded published aggregates; they are estimates rather than reported microdata. They are not medians or evidence that credit-line size causes default. An original score group is not equivalent to an observed future loss group.

The 30+, 60+ and 90+ delinquency measures are cumulative. They must satisfy `90+ ≤ 60+ ≤ 30+` within each denominator; adding them double counts accounts or balances. Comparing account-based and balance-based rates identifies differences in balance concentration, not a transition or roll rate. Credit-line increase and decrease shares do not establish net line-dollar changes or disjoint account populations.

The [variable definitions](https://www.philadelphiafed.org/surveys-and-data/y14-definitions) explain that net charge-off rates are annualized from quarterly net charge-offs using a three-month average balance. The pack retains the published rate. It does not multiply the rate by quarter-end balances and call the result observed dollar losses. Stocks such as balances are not summed into annual flow measures. The integrity checksum that sums a stock column is explicitly a software reconciliation, not an economic statistic.

## SQL and independent reconciliation

`market.sqlite` contains source-release provenance, a metric catalog, keyed observations and views `quarterly_source` and `quarterly_measures`. The observations have primary-key and foreign-key constraints. The pipeline runs population, missingness, cumulative-delinquency, percentile-order, complementary-group, denominator and SQLite integrity checks. Independent Python calculations check SQL aggregates and implied group averages.

Runnable query files are under `us_card_market/sql/`:

- `risk_report.sql`: levels and year-on-year rate/level comparisons using validated contiguous quarters.
- `origination_groups.sql`: below-660 and complementary original-score groups, with implied average opening limits.
- `reconciliations.sql`: population checks, visible nulls and aggregate reconstruction residuals.
- `quarterly_measures.sql`: the reusable derived-measure view used by the pipeline.

The first four quarters have no year-earlier comparator and retain null YoY values. Tests use clearly synthetic fixtures to challenge data contracts and arithmetic; the reference pack uses only the verified official files.

## Reading the risk pack

Start with `risk_indicators.csv` and `report.json`, then use `origination_groups.csv` to separate new-account mix from the amount of credit assigned. Check account-based versus balance-based delinquency and minimum-payment behavior. A useful investigation records an observed change, alternative explanations and the internal evidence needed to distinguish them—for example, account vintages, line-change histories, score migrations and product segmentation.

The aggregate data cannot establish why delinquency changed or whether a new policy worked. No arbitrary composite risk score, predictive forecast, account-level hypothesis test, or policy-profit claim is added. A genuine follow-up would require appropriately timed account outcomes and a defensible intervention design.

## Revisions and information timing

The [source methodology](https://www.philadelphiafed.org/surveys-and-data/y14-methodology) explains that reporting-panel entry/exit, confidentiality exclusions, extrapolation and resubmitted corrections can change historical observations. The pinned files are a **current published history**, not the values originally available at each historical date. The source publication date and actual retrieval timestamps are different provenance fields.

Re-running the same pinned cached bytes reproduces measures and checks. This does not turn the data into an original-vintage database. An online rolling forecast or prospective backtest on these historical values would require additional original-release evidence and is not claimed. A future revision comparison can retain a separately reviewed new release and compare identical quarter/metric keys; it must not fabricate older source snapshots.

## Attribution and source terms

Source: Federal Reserve Bank of Philadelphia, *Large Bank Credit Card and Mortgage Data*, 2026Q2 release, accessed October 2026. The source [terms of use](https://www.philadelphiafed.org/about-us/privacy-notice) permit informational, educational and research use and contain copyright and trademark conditions. **The software license does not relicense the source data or source-derived series.** Original downloads remain in an ignored local cache; generated research outputs carry a separate source-terms notice. No Federal Reserve logo, endorsement, affiliation or approval is implied. Conclusions are the project's own.
