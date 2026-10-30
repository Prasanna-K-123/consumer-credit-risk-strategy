-- Two complementary original-score groups inferred from published rounded aggregates.
-- These are arithmetic estimates, not reported microdata or medians.
SELECT quarter, '<660' AS original_score_group,
  new_accounts_mn*new_sub660_accounts_pct/100.0 AS estimated_new_accounts_mn,
  new_commitments_bn*new_sub660_commitments_pct/100.0 AS estimated_new_commitments_bn,
  new_sub660_avg_limit_usd AS implied_average_initial_limit_usd
FROM quarterly_measures
UNION ALL
SELECT quarter, '>=660',
  new_accounts_mn*(100.0-new_sub660_accounts_pct)/100.0,
  new_commitments_bn*(100.0-new_sub660_commitments_pct)/100.0,
  new_ge660_avg_limit_usd
FROM quarterly_measures
ORDER BY quarter,original_score_group;
