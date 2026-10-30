-- Each dataset should contain one row per quarter per declared metric.
SELECT o.dataset,COUNT(DISTINCT quarter) AS quarters,
  COUNT(*) AS observations, COUNT(DISTINCT metric) AS metrics,
  SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS null_observations
FROM observations o GROUP BY o.dataset;

-- Missing source observation is visible rather than imputed.
SELECT quarter,metric,value FROM observations WHERE value IS NULL;

-- Complementary inferred groups must reconcile back to totals, allowing floating point error.
WITH groups AS (
  SELECT quarter,
    new_accounts_mn*new_sub660_accounts_pct/100.0 +
      new_accounts_mn*(100.0-new_sub660_accounts_pct)/100.0 AS group_accounts_mn,
    new_commitments_bn*new_sub660_commitments_pct/100.0 +
      new_commitments_bn*(100.0-new_sub660_commitments_pct)/100.0 AS group_commitments_bn
  FROM quarterly_measures
)
SELECT q.quarter,g.group_accounts_mn-q.new_accounts_mn AS account_residual_mn,
  g.group_commitments_bn-q.new_commitments_bn AS commitment_residual_bn
FROM groups g JOIN quarterly_measures q USING(quarter)
WHERE ABS(g.group_accounts_mn-q.new_accounts_mn)>1e-9
   OR ABS(g.group_commitments_bn-q.new_commitments_bn)>1e-9;
