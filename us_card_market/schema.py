"""Explicit source contract. Percentage values retain percentage-point units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    key: str
    header: str
    unit: str
    style: str
    population: str
    aggregation: str
    is_share: bool = False


def m(key, header, unit, style, population="open non-charged-off consumer bankcards", aggregation="quarter-end stock", is_share=False):
    return Metric(key, header, unit, style, population, aggregation, is_share)


BALANCES = (
    m("balances_bn", "Total Balances ($Billions)", "USD billions", "currency"),
    m("accounts_mn", "Number of Accounts (Millions)", "million accounts", "number"),
    m("commitments_bn", "Total Commitments ($Billions)", "USD billions", "currency"),
    *(m(f"active_balance_p{p}_usd", f"Account Balance (Active Accounts Only) ({p}th percentile)", "USD", "currency", "active consumer bankcards", "account percentile") for p in (50, 75, 90)),
    *(m(f"limit_p{p}_usd", f"Current Credit Limit ({p}th Percentile)", "USD", "currency", aggregation="account percentile") for p in (50, 75, 90)),
    *(m(f"score_p{p}", f"Current Credit Score ({p}th percentile)", "credit score", "number", "accounts with eligible current credit score", "account percentile") for p in (10, 25, 50)),
    *(m(f"active_utilization_p{p}_pct", f"Utilization (Active Accounts Only) ({p}th percentile)", "percent", "percent", "active consumer bankcards", "account percentile") for p in (50, 75, 90)),
    m("line_decrease_pct", "Percentage of Accounts with Credit Line Decrease", "percent", "percent", aggregation="quarterly account share", is_share=True),
    m("line_increase_pct", "Percentage of Accounts with Credit Line Increase", "percent", "percent", aggregation="quarterly account share", is_share=True),
    *(m(f"dpd{d}_{basis}_pct", f"{d}+ Days Past Due Rates: {label} Based", "percent", "percent", aggregation=f"cumulative {d}+ days past due {basis} share", is_share=True) for d in (30, 60, 90) for basis, label in (("accounts", "Accounts"), ("balances", "Balances"))),
    m("minimum_payment_pct", "Share of Accounts Making the Minimum Payment", "percent", "percent", "active consumer bankcards", "last month of quarter account share", True),
    m("partial_payment_pct", "Share of Accounts Making Greater Than the Minimum Payment but Less Than the Full Balance", "percent", "percent", "active consumer bankcards", "last month of quarter account share", True),
    m("full_payment_pct", "Share of Accounts Making the Full Balance Payment", "percent", "percent", "active consumer bankcards", "last month of quarter account share", True),
    m("revolving_balances_bn", "Revolving Balances Only ($Billions)", "USD billions", "currency", "accounts revolving some or all of prior cycle balance"),
    m("nco_annualized_pct", "Net Charge-Off Rate", "annualized percent", "percent", aggregation="quarterly flow divided by three-month average balances, annualized"),
    m("purchase_apr_general_pct", "Average Purchase APR: General Purpose", "annual percent", "percent", "active general-purpose accounts with eligible non-promotional APR", "account average at quarter end"),
    m("purchase_apr_private_pct", "Average Purchase APR: Private Label", "annual percent", "percent", "active private-label accounts with eligible non-promotional APR", "account average at quarter end"),
    m("purchase_volume_bn", "Total Purchase Volume ($Billions)", "USD billions", "currency", aggregation="quarterly net-purchase flow"),
    *(m(f"purchase_avg_{key}_usd", f"Average Purchase Volume by Credit Score Group: {label} Credit Score", "USD per active account", "currency", f"active consumer bankcards with current credit score {label}", "quarterly net-purchase average") for key, label in (("sub660", "< 660"), ("660_719", "660-719"), ("ge720", ">= 720"))),
    m("promotional_apr_pct", "Percentage of Accounts Subject to Promotional APR: Percentage APR Accounts", "percent", "percent", "active consumer bankcards", "quarter-end account share", True),
)

ORIGINATIONS = (
    m("new_commitments_bn", "New Originations ($Billions)", "USD billions", "currency", "new consumer bankcard accounts", "quarterly initial-credit-limit flow"),
    m("new_accounts_mn", "Number of New Accounts (Millions)", "million accounts", "number", "new consumer bankcard accounts", "quarterly count"),
    *(m(f"new_limit_p{p}_usd", f"Original Credit Limit ({p}th percentile)", "USD", "currency", "new consumer bankcard accounts", "account percentile") for p in (50, 75, 90)),
    *(m(f"new_score_p{p}", f"Original Credit Score ({p}th percentile)", "credit score", "number", "new accounts with eligible origination credit score", "account percentile") for p in (10, 25, 50)),
    m("new_apr_general_pct", "Average Original Purchase APR: General Purpose", "annual percent", "percent", "new general-purpose bankcard accounts", "account average"),
    m("new_apr_private_pct", "Average Original Purchase APR: Private Label", "annual percent", "percent", "new private-label bankcard accounts", "account average"),
    *(m(f"new_limit_p50_{key}_usd", f"Median Original Credit Limit by Credit Score Group: {label} Credit Score", "USD", "currency", f"new consumer bankcards with original credit score {label}", "account percentile") for key, label in (("sub660", "< 660"), ("660_719", "660-719"), ("ge720", ">= 720"))),
    m("new_sub660_accounts_pct", "Percentage of New Accounts with < 660 Credit Score", "percent", "percent", "new consumer bankcard accounts", "quarterly account share", True),
    m("new_sub660_commitments_pct", "Percentage of New Commitments with < 660 Credit Score", "percent", "percent", "new consumer bankcard commitments", "quarterly commitment share", True),
)

SCHEMAS = {"balances": BALANCES, "originations": ORIGINATIONS}
FOOTER = (
    "",
    "Source: Philadelphia Fed FR Y-14M Data",
    "Website: https://www.philadelphiafed.org/surveys-and-data/large-bank-credit-card-and-mortgage-data",
    "Email: Phil.LargeBankData@phil.frb.org",
)
