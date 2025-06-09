"""All the general configuration of the project."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()

OVERLEAF_FIGURES = ROOT / "6839b3ec614c465dff006dc0" / "figures" / "autom_figures"

DOCUMENTS = ROOT.joinpath("documents").resolve()

TEMPLATE_GROUPS = ["marital_status", "highest_qualification"]


#-----------------
#Plot Settings

COLOR_SCHEME=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"]
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
sns.set_style("white")




