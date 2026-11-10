# Data sources

## `us_counties.csv.gz`

- **What:** balanced annual panel of 3,058 U.S. counties, 2000-2019 (61,160 rows).
- **Assembled by:** Augusto Cerqua, Marco Letta and Gabriele Pinto, for
  "On the (mis)use of machine learning with panel data", *Oxford Bulletin of
  Economics and Statistics*, 88(3), 506-518 (2026).
  <https://doi.org/10.1111/obes.70019>
- **Obtained from:** the authors' replication package,
  <https://github.com/gabrielepinto/results_ml_replication>, file
  `data/US_counties_dataset_2000-2019_ready_for_analysis.dta`, downloaded on
  2026-11-09.
- **Used in:** Wu and Xiao (2026), Section 6, arXiv:2610.21339.
- **Changes made:** 15 of the 28 original variables are kept and renamed; values
  are rounded to six decimals; nothing else is altered. The script that built the
  file is `make_package_data.py` in the project's working folder.

| Column | Original name | Unit |
|---|---|---|
| `county_id` | `ibrc_geo_id` | county FIPS code |
| `year` | `year` | calendar year |
| `state_fips` | `statefips` | state FIPS code |
| `county_name` | `county_name` | text |
| `income_pc` | `income_pc_` | personal income per capita, USD |
| `avg_wage` | `avg_wage_salaries_` | average wage per job, USD |
| `employment_rate` | `raw_employment_rate` | jobs located in the county per resident aged 18-65 |
| `population` | `pop_` | residents |
| `ui_share` | `perc_income_from_unempl_ins_` | % of personal income from unemployment insurance |
| `share_under18` | `perc_pop_under18_` | % of population |
| `share_over65` | `perc_pop_65plus_` | % of population |
| `births`, `deaths` | `births_per_1000ab`, `deaths_per_1000ab` | per 1,000 residents |
| `mig_international` | `saldo_intern_migrants_per_1000ab` | net, per 1,000 residents |
| `mig_domestic` | `saldo_domest_migrants_per_1000ab` | net, per 1,000 residents |

If you use these data, cite Cerqua, Letta and Pinto (2026).

## Simulated data

`simulate_panel` generates the Monte Carlo panel of Wu and Xiao (2026),
Appendix D.1 (baseline) and D.3 (alternative nuisance functions). Nothing is
stored on disk.
