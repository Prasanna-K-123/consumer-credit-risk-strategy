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

## Independent software checks

**What makes the SQL review useful?** Inspect the executable SQL population, confusion, calibration, segment and queue queries. Explain the join keys, uniqueness checks, window-function ordering and exact count reconciliation. A second metric implementation uses score groups to handle ranking ties and checks direct probability arithmetic.

**What was strengthened in the historical replay?** Input contracts reject invalid probabilities before clipping, fractional labels, mismatched shapes and malformed identities. Integrity failures use explicit exceptions that remain effective under optimized Python. Protocol, source and result identities are checked rather than trusting a success label.

**How do you reproduce the results?** Start with the root README commands, acquire the exact official source bytes, run the scientific replay, execute the SQL review, build the US market pack and generate the reports. Distinguish a successful current execution from historical receipts. Changed source bytes should stop reproduction and require a reviewed source update.

## Market reporting and judgment

**Why can account- and balance-based delinquency disagree?** They use different denominators. A small share of delinquent accounts can carry a larger share of balances. Compare rates only after checking the source's population and delinquency definition.

**What is a stock versus a flow?** Balances and commitments are quarter-end stocks; new accounts, initial credit commitments and purchases are quarterly flows. Net charge-offs are a quarterly flow rate annualized by the source. A ratio must respect those definitions.

**What does the market history prove?** It reproduces a particular release's retrieved, potentially revised series. It does not reconstruct every publication vintage or identify causal effects. Missing values remain missing, and source rounding limits precision of inferred ratios.

**What would be required for a bank decision?** Appropriate independent borrower histories, known feature availability, mature outcomes, validation across time and relevant segments, grounded cost/capacity constraints, model governance, security and measured intervention effects. A research repository demonstrates methods and judgment without implying those approvals already exist.
