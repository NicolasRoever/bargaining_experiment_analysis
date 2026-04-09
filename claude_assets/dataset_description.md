# Description of Cleaned Dataset

The cleaned dataset is located at `BLD / "data" / "merged_data_full_excluded.csv"`.
In the experiment, participants are paired with other participants and engage in a bargaining rounds (a total of 30 rounds). One row is a bargaining round for one participant. Since the participant is paired, you will find a second row with the same `negotiation_id` for the other participant in the pair.

The data containsthe following importamt variables:
- `negotiation_id`: the id of the negotiation pair. Two rows with the same `negotiation_id` belong to the same pair.
- `offer_time_XXX`: The time (in seconds from the beginning) at which the player made their offer number XXX. For example, `offer_time_1` is the time at which the player made their first offer, `offer_time_2` is the time at which the player made their second offer etc. If the player did not make a second offer, `offer_time_2` is NA.
- `offer_XXX`: The offer (in euros) the player made at their offer number XXX. For example, `offer_1` is the offer the player made at their first offer, `offer_2` is the offer the player made at their second offer etc. If the player did not make a second offer, `offer_2` is NA.
- `bargaining_outcome`: The outcome of the bargaining round. "Player" means a player terminated, "Random Termination" means the round was terminated randomly by the computer, "acceptance" means the players reached an agreement.
- `cumulated_TA_costs`: The cumulated time costs (in euros) the player had at the end of the round. 
- `information_asymmetry`: Whether the player was in the asymmetric information condition ("one-sided") or the symmetric information condition ("two-sided").
- `round`: The round number (4-33). Rounds 1-3 were practice rounds and are not included in the dataset.
- `participant_role`: Whether the player was a buyer or a seller in the round. Participants were randomly assigned to be buyers or sellers at the beginning of the experiment and stayed in this role for all rounds.
- `own_offer_accepted`: Whether the player's offer was accepted in the round by the other player
- `acceptance_time_sec`: The time (in seconds from the beginning of the round) at which an offer was accepted. If the round was terminated, this is NA.
- `termination_time_sec`: The time (in seconds from the beginning of the round) at which the round was terminated. If the round was accepted, this is NA.
-`gains_from_trade`: Buyer - seller valuation