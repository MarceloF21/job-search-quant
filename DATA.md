# Data sources and licences

| Data | Where it comes from | In this repo | Licence / terms |
|---|---|---|---|
| Toronto labour market (Sep–Oct 2026) | Employment and Social Development Canada, *Job Postings Advertised on Canada's National Job Bank Website*, one CSV per month: https://open.canada.ca/data/dataset/ea639e28-c0fc-48bf-b5dd-b8899bd43072 | Downloaded by `python -m job_search_quant labour` into the git-ignored `data/jobbank/` (about 50 MB per month). Only derived aggregates are committed: `docs/data/toronto_labour_2026-10.*` | Contains information licensed under the Open Government Licence – Canada (https://open.canada.ca/en/open-government-licence-canada) |
| Shortlist postings | Public job postings on employers' hiring systems and Job Bank, checked November 5–8, 2026 | `examples/sample_postings.csv`: short excerpts of pay, schedule and requirement lines, with links | Public postings, quoted briefly for analysis |
| Minimum wage | Government of Ontario, https://www.ontario.ca/minimumwage: $17.95 from October 1, 2026 | `job_search_quant/benchmarks.py` | Public information |
| GTA living wage | Ontario Living Wage Network, 2025 rates: $27.20 | `job_search_quant/benchmarks.py` | Public information |
| Applications journal | Your own records | `examples/sample_applications.csv` is **fictional**; a real journal belongs in the git-ignored `data/` | Private |

## How the open data is processed

- **City of Toronto:** rows with province Ontario and city Toronto, North York, Scarborough, Etobicoke, East York or York.
- **Hourly pay:**
  - Hourly postings use the midpoint of their pay range.
  - Annual, monthly, weekly and bi-weekly pay is divided by the posting's stated weekly hours, or by 40 when hours aren't stated.
  - Daily pay is divided by 8.
  - Values outside $10–$200/hour are dropped as data errors.
- **Entry level:** Job Bank's own postings (`External Indicator` = 0, which carry education and experience fields) that ask for high school or less and under a year of experience.
- **Minimum-wage bunching:** hourly-quoted postings whose minimum pay equals the old ($17.60) or new ($17.95) minimum, by the month the posting was first published.
