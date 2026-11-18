# Non-Destructive Corrosion Assessment of Reinforced Concrete Using Machine Learning and Multimodal Surface Data

Code and data for estimating **half-cell potential (HCP)** of reinforced concrete from four non-destructive inputs:

- surface resistivity (Proceq Resipod)
- a texture-based surface roughness index, extracted from surface photographs
- ambient temperature
- relative humidity

HCP indicates the *probability* of active reinforcement corrosion (ASTM C876). It does not measure corrosion rate or steel loss, so the models here estimate HCP and are intended as a screening tool.

**Authors:** Syeda Tamzida Akter, Niloy Roy, Sayed Shahriar Islam Lamun, Md. Basir Zisan
Department of Civil Engineering, Chittagong University of Engineering and Technology (CUET), Bangladesh

> The accompanying paper is under review. Citation details will be added on publication.

## Repository structure

```
data/
  data.csv                 Field dataset: 250 records from three RC buildings in Chattogram (237 with a roughness value)
  sample_images/           13 example concrete surface photographs used in the figures
notebooks/
  data_process_ML_results.ipynb      Preprocessing, 80/20 split, five regression models, reference-split results
  image_processing.ipynb             Sobel-gradient roughness extraction (grayscale, blur, gradient, heatmap)
  kde_plotting.ipynb                 Univariate distributions
  bivariate_analysis.ipynb           HCP vs each input
  pca_plotting.ipynb                 PCA
  figure_plotting.ipynb              Manuscript figures
  KNN_feature_vs_target.ipynb        KNN partial-dependence style plots
  updated_knn_target_feature_plot.ipynb
  early_exploration_colab.ipynb      Early Colab experiments (outputs removed to keep the file small)
scripts/
  revision_analysis.py     30 repeated random splits, feature-set comparison, leave-one-session-out test
  make_figures.py          Model-stability and feature-set figures from the outputs of revision_analysis.py
  fix_roughness_figs.py    Roughness KDE and HCP-vs-roughness figures on the true index scale
app/
  app.py                   Gradio web interface: upload a surface photo, enter resistivity, temperature and humidity, get predicted HCP
figures/                   Results figures from the repeated-split analysis
```

## Dataset

`data/data.csv` has one row per measurement point:

| Column | Unit | Description |
|---|---|---|
| `temperature_c` | °C | Ambient temperature (one value per measurement session) |
| `humidity_%` | % | Relative humidity (one value per measurement session) |
| `half-cell_potential_v` | V vs CSE | Measured half-cell potential (target) |
| `surface_resistivity` | kΩ·cm | Wenner four-probe surface resistivity |
| `surface_roughness` | – | Standard deviation of the 8-bit Sobel gradient magnitude of the surface image |

Thirteen rows have no roughness value and are dropped before modelling, leaving 237 samples. Temperature and humidity were recorded once per inspection session, so the 237 samples share 14 distinct temperature–humidity pairs.

## Roughness index

For each surface photograph (see `notebooks/image_processing.ipynb`):

1. Convert to grayscale.
2. Apply a 5 × 5 Gaussian blur.
3. Compute horizontal and vertical Sobel gradients (3 × 3).
4. Combine them into the gradient magnitude √(Gx² + Gy²) and scale it to 8-bit (0–255).
5. Take the standard deviation of all pixels as the roughness index.

## Main results (237 samples, 30 random 80/20 splits, all four inputs)

| Model | R² (mean ± SD) | RMSE (V) | NRMSE (% of HCP range) |
|---|---|---|---|
| Random Forest | 0.752 ± 0.077 | 0.064 ± 0.013 | 11.2 ± 2.3 |
| KNN | 0.689 ± 0.157 | 0.070 ± 0.010 | 12.2 ± 1.7 |
| Bayesian Ridge | 0.600 ± 0.079 | 0.082 ± 0.009 | 14.1 ± 1.6 |
| Linear Regression | 0.600 ± 0.082 | 0.082 ± 0.009 | 14.2 ± 1.6 |
| SVR | 0.530 ± 0.225 | 0.086 ± 0.015 | 14.9 ± 2.6 |

Random Forest R² by input set: resistivity only 0.49; + roughness 0.65; + temperature and humidity 0.74; all four 0.75. When a whole measurement session is held out, R² falls to about 0.35, so the models are calibrated to the surveyed structures and need local data before use elsewhere.

## How to run

```bash
pip install -r requirements.txt

# Reproduce the repeated-split, feature-set and session hold-out analysis (about 15 minutes on a laptop)
mkdir -p out
python scripts/revision_analysis.py data/data.csv out
python scripts/make_figures.py out

# Launch the web interface
cd app
pip install -r requirements.txt
python app.py
```

The notebooks were written on Windows. A few cells load the Helvetica font from `C:\Users\...` or read images from `F:\...`. Change those paths to your own, or delete the font cells to use matplotlib's default font.

## Contact

Syeda Tamzida Akter (corresponding code author) · Md. Basir Zisan (corresponding author), CUET
