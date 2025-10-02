# Templates for Reproducible Research Projects in Economics


- We should ask about subjective time pressure in the treatments!
- We could also ask about subjective bargaining power
- We could ask if people think this is a repeated game or not, i.e. better get at social punishment. 

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


### Plan for Today 

- Add Split gains from trade in Appendix table tab:regressions_asymmetric_bargaining
- Fix Cox curves
- Evidence on Division
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

- We have a negotiation in the first session, where acceptance time raw has the correct seconds. How is that possible>? 


## On TA-Cost Differences
- We currently have some observations in the first four data collections which have widely differing TA costs (hinting at the bigger problem with how we record time and try to fix it afterwards!); My current judgement is that we need to improve our time corrections for these observations, and then we should be fine (we are fine for outcomes of gains_from_trade anyway)

### Explanation Cases lsat_offer_time > bargaining_time_full

In some cases, the computer/other player is terminating as a player is submitting an offer. In this case, the offer time can be larger than the termination time, because of latency between the client and the server---the client has not been notified that the negotiation has been terminated. This happens in very few cases though, and the average difference between last offer time and termination time is 0.1 seconds; this shows that our application has very low latency. We do not consider this feature of our data to be a problem for data quality. 

## Drop Outs

- In session with ID '8jp2clvt', player with ID "dE5arGFL" dropped out after round 26 (for some reason, he came back later to fill in demogrphics, so these we can use, but rounds 27-30 are discarded); he is group_in_session_4, and player with ID lbnJrtKO dropped out after round 10

## Before Publishing the Paper
- Check that number of negotiations is round



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

## Dataset: `merged_data_full_excluded.csv`

The cleaned dataset used throughout the analysis is stored in `bld/data/merged_data_full_excluded.csv` and assembled in
`src/bargaining_analysis/clean_data/functions_clean_data.py`. The file contains one row per participant per bargaining round
after applying the following exclusion steps:

1. **Pre-registered exclusions** (`apply_preregistered_exclusion_criteria`)
   - Remove observations flagged as `mistake` where `payoff` is smaller than -15. 
   - Drop the first round of real bargaining
2. **Technical exclusions** (`apply_technical_exclusion_criteria`)
   - Exclude negotiations with a difference of more than four seconds between `current_second` and `bargaining_time_full_sec`.
   - Exclude negotiations where both acceptance and termination times were recorded.


### Variables

Below is a short description of every column in
`merged_data_full_excluded.csv`.
Dynamic sequences of offers and times (`offer_1`, `offer_2`, … and
`offer_time_1`, `offer_time_2`, …) follow the same naming logic and are
abbreviated here.

- `participant_id_in_session` – unique identifier for each participant
  within a session.
- `participant_label` – unique identifier for each participant in our experiment.
- `participant_code` – random code assigned by oTree.
- `participant_role` – either `Buyer` or `Seller`.
- `group_id_in_round` – group identifier within a round.
- `round` – bargaining round number.
- `session_id` – identifier of the experimental session (we run sessions with 32 subjects each)
- `information_asymmetry` – whether valuations of only the buyer were private
  (`one-sided`) or symmetric (`two-sided`).
- `TA_costs` – transaction costs per second.
- `treatment` – shorthand combining information asymmetry and costs (T1/T2/T3/T4).
- `subsession.is_practice_round` – indicator for practice rounds.
- `Role_Seller` – equals `1` for sellers and `0` for buyers.
- `id_in_group` – player's id within the group.
- `cumulated_TA_costs` – total transaction costs incurred by the player.
- `bargaining_outcome` – outcome of the negotiation (`acceptance` or
  type of termination).
- `termination_mode` – specific termination reason if no agreement was
  reached.
- `accepted_by_id_in_group` – id of the player whose offer was
  accepted.
- `agreement_dummy` – equals `1` if the bargaining ended in acceptance.
- `own_offer_accepted` – equals `1` if the player's own offer was
  accepted.
- `terminated_by_id_in_group` – id of the player who terminated the
  negotiation.
- `player_terminated` – equals `1` if the player actively terminated.
- `bargain_start_time_unix` – unix timestamp when the round started.
- `acceptance_time_raw` – raw acceptance timestamp.
- `acceptance_time_1000_adj` – acceptance time adjusted for the
  millisecond bug.
- `termination_time_raw` – raw termination timestamp.
- `termination_time_1000_adj` – termination time adjusted for the bug.
- `client_time_correction` – offset between client and server times.
- `acceptance_time_sec` – corrected acceptance time in seconds.
- `termination_time_sec` – corrected termination time in seconds.
- `bargaining_time_full_sec` – full bargaining duration in seconds.
- `current_second` – last recorded second for the player.
- `offer_1`, `offer_2`, … – sequence of offers made by the player.
- `offer_time_1`, `offer_time_2`, … – times of those offers in seconds.
- `last_offer` – final price offered by the player.
- `last_offer_time` – time of the final offer.
- `number_of_offers` – how many offers the player made.
- `deal_price_raw` – raw price at which the good traded.
- `deal_price` – corrected deal price used for payoffs.
- `valuation` – player's private valuation of the good.
- `payoff` – payoff in currency units after accounting for costs.
- `relative_valuation` – valuation rescaled by treatment as described in
  the code.
- `valuation_bucket` – categorical valuation: Low (<7.72), Medium
  (7.72–21.40) or High (>21.40).
- `gains_from_trade` – buyer valuation minus seller valuation within the
  pair.
- `split_gains_from_trade` – player's share of the gains from trade.
- `gains_from_trade_dummy` – equals `1` if gains from trade are
  non‑negative.
- `large_gains_from_trade_indicator` – equals `1` if gains from trade
  are at least `20`.
- `majority_gains_from_trade_indicator` – equals `1` if the player's
  share of gains from trade is at least `50%`.
- `positive_gains_symmetric_treatment` – indicator for positive gains in
  the two‑sided treatments.
- `small_gains_from_trade_indicator` – equals `1` if gains from trade
  lie between `0` and `10`.
- `first_offer` – equals `1` for the player who made the first offer.
- `first_offer_split` – first offer expressed as a fraction of the gains
  from trade.
- `efficiency` – `1` if the outcome is efficient given the gains from
  trade, `0` otherwise.
- `seller_info_public` – equals `1` for sellers in the one‑sided
  treatment.
- `ultimatum_offer` – ultimatum offer entered in the final stage.
- `ultimatum_indicator` – equals `1` if the ultimatum offer was
  ≤ 50% of the pie.
- `risk_elicitation_choice` – choice in the risk elicitation task.
- `time_row_1`–`time_row_6` – answers in the time preference task.
- `time_preference_switching_points` – index of the first time‑preference
  row with a delayed payout.
- `experiment_start_time` – timestamp of the session start.
- `experiment_end_time` – timestamp of the session end.
- `experiment_duration` – total session duration in seconds.
- `age` – age of the participant.
- `gender` – participant's stated gender.
- `strategy_answer` – free‑text answer describing bargaining strategy.
- `mistake` – equals `1` if `-cumulated_TA_costs` exceeded `payoff`.
- `negotiation_id` – unique id for each pair in each round.
- `time_inconsistency_dummy` – equals `1` for negotiations with timing
  inconsistencies.
- `group_id_in_session` – id linking pairs across rounds within a
  session.
- `group_id` – global group identifier across sessions.

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
