-- Percent inputs are 0-to-100 values. USD billions / million accounts * 1000 = USD/account.
CREATE VIEW quarterly_measures AS
SELECT *,
  1000.0 * balances_bn / accounts_mn AS avg_balance_per_account_usd,
  100.0 * balances_bn / commitments_bn AS aggregate_balance_to_commitment_pct,
  100.0 * revolving_balances_bn / balances_bn AS revolving_to_total_balance_pct,
  100.0 - minimum_payment_pct - partial_payment_pct - full_payment_pct AS unclassified_payment_share_pct,
  1000.0 * new_commitments_bn / new_accounts_mn AS new_avg_initial_limit_usd,
  1000.0 * new_commitments_bn * new_sub660_commitments_pct /
    NULLIF(new_accounts_mn * new_sub660_accounts_pct, 0) AS new_sub660_avg_limit_usd,
  1000.0 * new_commitments_bn * (100.0-new_sub660_commitments_pct) /
    NULLIF(new_accounts_mn * (100.0-new_sub660_accounts_pct), 0) AS new_ge660_avg_limit_usd,
  new_sub660_accounts_pct-new_sub660_commitments_pct AS new_sub660_mix_gap_pp
FROM quarterly_source;
