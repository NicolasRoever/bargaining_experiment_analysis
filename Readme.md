# Bargaining Experiment Analysis


TO push to overleaf:



```
git fetch overleaf
git checkout -b overleaf-export overleaf/master
git checkout main -- 6839b3ec614c465dff006dc0/
git add .
git commit -m "📄 Overleaf export: keep only TeX, Bib, figures"
git push overleaf overleaf-export:master
```

```
git checkout main

git checkout --orphan overleaf-export

git rm -rf .

git checkout main -- 6839b3ec614c465dff006dc0/

git add .

git commit -m "📄 Overleaf export"

git push -f overleaf overleaf-export:master

```
