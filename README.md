# Leave-one-site-out autism classification on ABIDE I

Course project. The question is whether a resting-state connectivity model still separates autism from control at a scanner it was not trained on.

This is a research analysis of a public, de-identified release. It is not a diagnostic test.

## Result in one paragraph

On 856 quality-checked ABIDE I scans (398 autism, 458 control, 20 site labels), a logistic regression on Fisher-z correlations among 200 regions has macro leave-one-site-out balanced accuracy **0.67** and AUROC **0.74**. Age and sex sit at **0.52**. Pooled cross-validation does not beat leave-one-site-out. MaxMun is **0.54**; USM, the strongest larger site, is **0.79**. The same features identify the acquisition site well above the majority-site base rate. Probabilities are only slightly sharper than the base rate (Brier **0.21** versus **0.25**).

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/abide_study.py
```

The first run downloads the CPAC, band-pass, no-global-signal, Craddock-200 time series from the Preprocessed Connectomes Project (about 871 files). Later runs reuse `data/nilearn_data/features.npz`. Raw downloads and the virtualenv are gitignored.

The script writes `results/summary.json`, `results/per_site.csv`, and the figures under `results/` and `report/figures/`. Open `notebooks/analysis.ipynb` from `notebooks/` after that run so the relative paths resolve.

## Data

ABIDE I phenotypic and imaging data were shared by the contributing sites. This project uses the CPAC preprocessed derivatives from the Preprocessed Connectomes Project, filtered to subjects who passed every quality-control rater, with band-pass filtering and without global signal regression. Cite Di Martino et al., Molecular Psychiatry, 2014, and the preprocessing release. Do not attempt to re-identify anyone.
