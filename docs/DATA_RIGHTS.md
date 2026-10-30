# Sources and attribution

## Card-client study

I-Cheng Yeh, *Default of Credit Card Clients*, UCI Machine Learning Repository,
dataset 350, DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
[Official source and license](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
The source is licensed CC BY 4.0. Source-derived card research tables and
predictions retain this attribution and license. Changes include grouped sample
partitions, model predictions, calibration and decision-review diagnostics.
The original source workbook is acquired into the ignored local cache.

The records describe existing Taiwanese card customers in 2005. They do not
describe US applicants. The original dataset's identifiers are source-row
identifiers; they are not an identity registry or current customer accounts.

## US bankcard market series

Federal Reserve Bank of Philadelphia, *Large Bank Credit Card and Mortgage Data*,
2026 Q2 release, balance and origination series.
[Official data](https://www.philadelphiafed.org/surveys-and-data/large-bank-credit-card-and-mortgage-data)
and [definitions and FAQs](https://www.philadelphiafed.org/surveys-and-data/y14-frequently-asked-questions).

The project's software license does not relicense Federal Reserve source material.
Source CSVs are downloaded from official release URLs into an ignored cache;
the repository provides source identities, attributed analytical summaries and
transformations. Users should consult the source's terms of use. US aggregate
records are never joined to individual Taiwanese records or described as
account-level model validation.

## Software and study history

This repository contains an October 2026 software extension and a preserved
historical card experiment. See [CARD_HISTORY](CARD_HISTORY.md) for the exact
historical source and evidence identities and the limits of the original
recorded protocol. A current software release does not turn that historical
evaluation into a new unseen test.
