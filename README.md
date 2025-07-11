# Templates for Reproducible Research Projects in Economics

```bash
 export PYTHONPATH=$PYTHONPATH:/Users/nicolasroever/Dropbox/Promotion/Bargaining/bargaining_experiment_analysis/src
```

Overleaf workflow:
```bash
cd overleaf-docs
git add .
git commit -m "Update"
git pull
git push   
cd ..       
```


### Open To-Do

- Fix how we correct times in the first 4 sessions; there are still issues.
- Fix agreement rates filter (larger not larger equal 0. )
- Fix number of offers function (wrong somehow : (.)
- Check how you adjusted times more thoroughly!
- Participant with label 72TouSYb still has weird first offer times. We need to fix this somehow!
- dwjn1hbc has negative first offer time, should exclude him as well


### Errata

- In the first 4 sessions, we recorded offer times, acceptance times and player termination times on the client, leading to in some cases large measurement errors caused by differences in time between the client and the server. In cases with transaction costs, I fixed this, in the case of no transaction costs, we are screwed!

- The accepted deal price is formatted by a sub function in the client in the first 4 sessions, leading to * 10 formatting error in few cases.  

- We have an issue in a negotiation in T2, both have different bargaining_times for the same negotiation. My most likely explanation is internet issues for one person. Need to adress this later! (could just compute TA_costs for Random_Termination based on the termination times)

- Check negotiation ID 454, somehow we have both acceptance and terminatnion (internet issues?)


## On TA-Cost Differences
- We currently have some observations in the first four data collections which have widely differing TA costs (hinting at the bigger problem with how we record time and try to fix it afterwards!); My current judgement is that we need to improve our time corrections for these observations, and then we should be fine (we are fine for outcomes of gains_from_trade anyway)

### Explanation Cases lsat_offer_time > bargaining_time_full

In some cases, the computer/other player is terminating as a player is submitting an offer. In this case, the offer time can be larger than the termination time, because of latency between the client and the server---the client has not been notified that the negotiation has been terminated. This happens in very few cases though, and the average difference between last offer time and termination time is 0.1 seconds; this shows that our application has very low latency. We do not consider this feature of our data to be a problem for data quality. 




![MIT license](https://img.shields.io/github/license/OpenSourceEconomics/econ-project-templates)
[![image](https://zenodo.org/badge/14557543.svg)](https://zenodo.org/badge/latestdoi/14557543)
[![Documentation Status](https://readthedocs.org/projects/econ-project-templates/badge/?version=stable)](https://econ-project-templates.readthedocs.io/en/stable/)
[![image](https://github.com/OpenSourceEconomics/econ-project-templates/actions/workflows/main.yml/badge.svg)](https://github.com/OpenSourceEconomics/econ-project-templates/actions/workflows/main.yml)
[![image](https://codecov.io/gh/OpenSourceEconomics/econ-project-templates/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenSourceEconomics/econ-project-templates)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/OpenSourceEconomics/econ-project-templates/main.svg)](https://results.pre-commit.ci/latest/github/OpenSourceEconomics/econ-project-templates/main)

This project provides a template for economists aimed at facilitating the production of
reproducible research using the most commonly used programming languages in the field,
such as Python, R, Julia, and Stata.

> [!NOTE]
> Although the underlying architecture supports all listed programming languages, the
> current template implementation is limited to Python and R.

## Getting Started

You can find all necessary resources to get started on our
[documentation](https://econ-project-templates.readthedocs.io/en/stable/).

## Contributing

We welcome suggestions on anything from improving the documentation to reporting bugs
and requesting new features. Please open an
[issue](https://github.com/OpenSourceEconomics/econ-project-templates/issues) in these
cases.

If you want to work on a specific feature, we are more than happy to get you started!
Please [get in touch briefly](https://www.wiwi.uni-bonn.de/gaudecker), this is a small
team so there is no need for a detailed formal process.

### Contributors

@hmgaudecker @timmens @tobiasraabe @mj023

### Former Contributors

@janosg @PKEuS @philippmuller @julienschat @raholler
