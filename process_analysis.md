# Analysis of Bargaining Process

To analyze the bargaining process, we do the following: 

1.  we focus on multi-offer rounds,
that is, negotiations in which at least two offers are submitted.

2. For each participant-round, we normalize offer timing to the unit interval [0, 1], where 0 denotes the participant’s first offer and 1
denotes the participant’s last offer in that round. We then divide normalized time into 15 equal-width bins. Within each treatment, role, and bin, we compute the mean submitted offer

3. Produce a figure, which shows the resulting convergence gap, defined as the bin-level mean seller offer minus the bin-level mean
buyer offer

4. Produce the following plot: For each offer
event within a match, let the midpoint be the average of the player’s own most recent offer and the
opponent’s most recent offer. We then normalize time at the match level so that both players share
a common within-round clock, divide this clock into 15 equal-width bins, and compute the mean
current offer and the mean midpoint benchmark separately by treatment and role. We exclude initial
offers for which the opponent has not yet made any proposal, since the midpoint is not defined for
those events. Only produce this figure for BuyerCost.