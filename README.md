# Bargaining Experiment Anbalysis



## Drop Outs

- In session with ID '8jp2clvt', player with ID "dE5arGFL" dropped out after round 26 (for some reason, he came back later to fill in demogrphics, so these we can use, but rounds 27-30 are discarded); he is group_in_session_4, 

- player with ID lbnJrtKO dropped out after round 10

- One person did not fill in demographics, but we can still use their bargaining data


![MIT license](https://img.shields.io/github/license/OpenSourceEconomics/econ-project-templates)
[![image](https://zenodo.org/badge/14557543.svg)](https://zenodo.org/badge/latestdoi/14557543)
[![Documentation Status](https://readthedocs.org/projects/econ-project-templates/badge/?version=stable)](https://econ-project-templates.readthedocs.io/en/stable/)
[![image](https://github.com/OpenSourceEconomics/econ-project-templates/actions/workflows/main.yml/badge.svg)](https://github.com/OpenSourceEconomics/econ-project-templates/actions/workflows/main.yml)
[![image](https://codecov.io/gh/OpenSourceEconomics/econ-project-templates/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenSourceEconomics/econ-project-templates)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/OpenSourceEconomics/econ-project-templates/main.svg)](https://results.pre-commit.ci/latest/github/OpenSourceEconomics/econ-project-templates/main)



## Technical Notes for Developers
```bash
 export PYTHONPATH=$PYTHONPATH:/Users/nicolasroever/Dropbox/Promotion/Bargaining/bargaining_experiment_analysis/src
```

Overleaf workflow:
```bash
cd overleaf-docs
git add .
git commit -m "Update"
git pull
git push origin HEAD:master
cd ..       
```

- Investigate ID n3uicero in session with ID '76kuqtpi'; there was an earlier note that this person might have dropped out, but they have data for all rounds, so we can use it; they also filled in demographics, so my judgement is we use that as well.