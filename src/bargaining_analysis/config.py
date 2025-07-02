"""All the general configuration of the project."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()

OVERLEAF_FIGURES = ROOT.joinpath("overleaf-docs", "figures", "autom_figures").resolve()
OVERLEAF_TABLES = ROOT.joinpath("overleaf-docs", "tables", "autom_tables").resolve()

BLD = ROOT.joinpath("bld").resolve()

DOCUMENTS = ROOT.joinpath("documents").resolve()

TEMPLATE_GROUPS = ["marital_status", "highest_qualification"]


#-----------------
#Plot Settings

COLOR_SCHEME=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"]
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
sns.set_style("white")

#Legend Elements
LEGEND_ELEMENTS_OUTCOME = [
    mlines.Line2D([], [], color=COLOR_SCHEME[0], marker='o', linestyle='None', label='Player Termination'),
    mlines.Line2D([], [], color=COLOR_SCHEME[1], marker='o', linestyle='None', label='Computer Termination'),
    mlines.Line2D([], [], color=COLOR_SCHEME[2], marker='o', linestyle='None', label='Acceptance')
]
#Regression Stuff
VARLABELS_REGRESSION = {'const':'Constant','first_offer':'First Offer', 'split_gains_from_trade':'Split Gains from Trade', 'efficiency':'Efficiency', 'agreement_dummy':'Dummy Agreement', 'C(participant_role)[T.Seller]':'Seller', 'C(TA_costs)[T.0.05]':'TA Costs $= 0.05$', 'gains_from_trade':'Gains from Trade', 
    "C(information_asymmetry)[one-sided]:small_gains_from_trade_indicator": "One-sided $\\times$ Small Gains from Trade",
    "C(information_asymmetry)[two-sided]:small_gains_from_trade_indicator": "Two-sided $\\times$ Small Gains from Trade",
    "C(information_asymmetry)[one-sided]:positive_gains_symmetric_treatment": "One-sided $\\times$ Positive Gains from Trade",
    "C(information_asymmetry)[two-sided]:positive_gains_symmetric_treatment": "Two-sided $\\times$ Positive Gains from Trade",
    "C(information_asymmetry)[one-sided]:ultimatum_indicator": "One-sided $\\times$ Ultimatum Indicator",
    "C(information_asymmetry)[two-sided]:ultimatum_indicator": "Two-sided $\\times$ Ultimatum Indicator",
    "C(information_asymmetry)[T.two-sided]": "Symmetric Uncertainty", 
    "positive_gains_symmetric_treatment": "Positive Gains from Trade $\\times$ Symmetric Uncertainty",
    "ultimatum_indicator": "Ultimatum Offer $>50\%$",
    "risk_elicitation_choice": "Risk Aversion",
    "time_preference_switching_points": "Time Preference",
    "bargaining_time_full_sec": "Bargaining Time (sec)",
    "first_offer_split": "First Offer \\% of Gains from Trade",
    "C(information_asymmetry)[T.two-sided]:C(TA_costs)[0.0]": "Symmetric Uncertainty $\\times$ TA Costs $= 0.0$",
    "C(information_asymmetry)[T.two-sided]:C(TA_costs)[0.05]": "Symmetric Uncertainty $\\times$ TA Costs $= 0.05$",
    "C(participant_role)[Buyer]:valuation": "Buyer $\\times$ Valuation",
    "C(participant_role)[Seller]:valuation": "Seller $\\times$ Valuation",
    "C(participant_role)[T.Seller]:C(TA_costs)[0.0]": "Seller $\\times$ TA Costs $= 0.0$",
    "C(participant_role)[T.Seller]:C(TA_costs)[0.05]": "Seller $\\times$ TA Costs $= 0.05$",
    "Intercept": "Constant",
    "small_gains_from_trade_indicator" : "Gains from Trade $\\leq 10",
    }




