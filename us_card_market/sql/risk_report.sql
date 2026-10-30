-- Run with sqlite3 reference/us_market/market.sqlite < us_card_market/sql/risk_report.sql
-- Contiguous quarters are validated before these LAG offsets are used.
SELECT quarter,
  balances_bn,
  ROUND(100.0*(balances_bn / LAG(balances_bn,4) OVER w - 1),6) AS balance_yoy_growth_pct,
  accounts_mn,
  ROUND(100.0*(accounts_mn / LAG(accounts_mn,4) OVER w - 1),6) AS accounts_yoy_growth_pct,
  commitments_bn,
  aggregate_balance_to_commitment_pct,
  dpd90_accounts_pct,
  dpd90_balances_pct,
  ROUND(dpd90_balances_pct-LAG(dpd90_balances_pct,4) OVER w,6) AS dpd90_balance_yoy_change_pp,
  nco_annualized_pct,
  ROUND(nco_annualized_pct-LAG(nco_annualized_pct,4) OVER w,6) AS nco_yoy_change_pp,
  minimum_payment_pct,
  ROUND(minimum_payment_pct-LAG(minimum_payment_pct,4) OVER w,6) AS minimum_payment_yoy_change_pp,
  new_accounts_mn,
  new_commitments_bn,
  new_sub660_accounts_pct,
  ROUND(new_sub660_accounts_pct-LAG(new_sub660_accounts_pct,4) OVER w,6) AS sub660_account_yoy_change_pp,
  new_sub660_commitments_pct,
  new_sub660_avg_limit_usd,
  new_ge660_avg_limit_usd
FROM quarterly_measures
WINDOW w AS (ORDER BY quarter_index)
ORDER BY quarter_index;
