# Ground-Level Ozone Photochemistry, Statistical Distribution Tail Modeling, and Meteorological Machine Learning Generalization across an Urban-Industrial Megacity Airshed

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Target: Atmospheric Environment](https://img.shields.io/badge/Journal-Atmospheric%20Environment%20(Elsevier%2C%20Q1)-0073B1.svg)](https://www.sciencedirect.com/journal/atmospheric-environment)

This repository contains the complete open-source reproducibility pipelines, quality-controlled datasets, statistical distribution fitting algorithms, and machine learning forward-transfer architectures accompanying the research manuscript:

> **"Ground-Level Ozone Photochemistry, Statistical Distribution Tail Modeling, and Meteorological Machine Learning Generalization across an Urban-Industrial Megacity Airshed"**  
> **Author:** Saptarshi Sadhu (Sole Author)  
> **Affiliation:** School of Engineering and Technology, Adamas University, Kolkata 700126, India  
> **Correspondence:** `saptarshi1.sadhu@stu.adamasuniversity.ac.in` | `saptarshisadhuofficial@gmail.com`  
> **Target Journal:** *Atmospheric Environment* (Elsevier, Q1, Impact Factor: ~4.8–5.4)

---

## 📌 Research Overview

Ground-level ozone ($\\mathrm{O_3}$) in tropical urban airsheds exhibits non-linear dependencies on volatile organic compound (VOC) and nitrogen oxide ($\\mathrm{NO_x}$) precursors. This investigation establishes a continuous empirical investigation across all 13 Continuous Ambient Air Quality Monitoring System (CAAQMS) stations in the Kolkata–Howrah conurbation during 2024–2025 (**16.1 million continuous raw parameter measurements**).

### Core Scientific Contributions:
1. **Empirical Photochemical Regimes:** Station-wise classification of precursor indicator ratios ($\\sum\\mathrm{BTEX}/\\mathrm{NO}_x$, midday $\\mathrm{NO}_2/\\mathrm{NO}$, and total oxidant $[\\mathrm{O}_x] = [\\mathrm{O}_3] + [\\mathrm{NO}_2]$), delineating traffic-dominated titration sinks from VOC-sensitive industrial receptor plumes.
2. **182-Series Maximum Likelihood Tail Modeling:** Rigorous evaluation of two-parameter Lognormal, Gamma, and Weibull distributions (`floc=0`) across 182 spatiotemporal criteria pollutant series (7 pollutants $\\times$ 13 stations $\\times$ 2 years; 5,048,133 positive 15-minute observations). The Lognormal distribution achieved AIC optimality in $>81\\%$ of series, refuting the universal applicability of Weibull models and demonstrating that atmospheric dispersion in this tropical airshed is governed by Kolmogorov–Ott multiplicative dilution processes.
3. **Cross-Year Machine Learning Generalization:** Evaluates forward-transfer generalization using Random Forest regression trained on 2024 hourly meteorology and tested on unseen 2025 observations across all 13 stations. Suburban and background sites achieve robust generalization ($R^2 = 0.50\\text{--}0.68$), whereas traffic corridors collapse to negative skill ($R^2 = -0.26$, $\\text{RMSE} = 13.83\\,\\mu\\text{g m}^{-3}$).
4. **Diagnostic Chemical Counter-Experiments:** Proves via diurnal odd-oxygen conservation ($[\\mathrm{O}_x]$) that supplying co-located $\\mathrm{NO}_x$ and VOC measurements recovers positive forward transfer skill ($R^2 = +0.02$) and cross-validated accuracy ($R^2 = 0.9533$), confirming that the generalization breakdown is a chemically driven artifact of localized rush-hour nitric oxide titration.

---

## 🗺️ 13 CAAQMS Stations & Photochemical Regime Indicators

| Station Name | Code | Zone Classification | $\\sum\\text{BTEX}$ ($\\mu\\text{g m}^{-3}$) | $\\text{NO}_x$ (ppb) | $\\sum\\text{BTEX}/\\text{NO}_x$ | Midday $\\text{NO}_2/\\text{NO}$ | $[\\text{O}_x]$ ($\\mu\\text{g m}^{-3}$) | ML Transfer $R^2$ |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Barrackpore** | SVSPA | Industrial/Suburban | 13.6 | 17.0 | 0.803 | 2.55 | 31.8 $\\pm$ 19.2 | **0.6825** |
| **Victoria Memorial** | VC | Urban Background | 9.1 | 25.7 | 0.352 | 1.50 | 46.5 $\\pm$ 27.0 | **0.5914** |
| **Rabindra Sarovar** | RS | Urban Park | 12.5 | 42.1 | 0.296 | 2.13 | 73.6 $\\pm$ 42.6 | **0.5008** |
| **Padmapukur** | PP | Industrial/Residential | 15.7 | 28.1 | 0.558 | 1.47 | 48.0 $\\pm$ 25.3 | **0.4865** |
| **Jadavpur** | JD | Mixed Urban | 29.5 | 42.7 | 0.691 | 1.29 | 52.6 $\\pm$ 30.7 | **0.4712** |
| **Dasnagar** | DN | Heavy Industrial | 20.3 | 34.7 | 0.584 | 3.91 | 88.3 $\\pm$ 44.3 | **0.4237** |
| **Fort William** | FW | Semi-Urban/Military | 8.9 | 36.4 | 0.245 | 2.44 | 74.1 $\\pm$ 36.5 | 0.2880 |
| **Ghusuri** | GH | Heavy Industrial | 18.4 | 25.3 | 0.728 | 2.41 | 37.9 $\\pm$ 23.4 | 0.0714 |
| **Rabindra Bharati Univ** | RBU | Traffic Hotspot | 16.3 | 36.6 | 0.446 | 2.05 | 59.8 $\\pm$ 33.2 | 0.0755 |
| **Botanical Garden** | BOG | Green Background | 34.3 | 11.5 | 2.984 | 5.71 | 40.5 $\\pm$ 22.8 | 0.0499 |
| **Belur Math** | BM | Religious/Mixed | 13.4 | 33.8 | 0.397 | 2.09 | 52.3 $\\pm$ 30.9 | 0.0175 |
| **Ballygunge** | BG | Residential Traffic | 24.3 | 44.8 | 0.542 | 2.88 | 57.3 $\\pm$ 33.1 | **-0.1722** |
| **Bidhannagar** | BN | Commercial Corridor | 12.2 | 30.5 | 0.400 | 1.43 | 48.6 $\\pm$ 26.7 | **-0.2577** |

---

## 📁 Repository Directory Structure

```text
kolkata-howrah-ozone-photochemistry/
├── README.md                                          <-- Master Documentation
├── LICENSE                                            <-- MIT License
├── requirements.txt                                   <-- Package dependencies
├── CITATION.cff                                       <-- Citation metadata
├── .gitignore                                         <-- Git exclusion rules
│
├── data/
│   ├── master_cleaned.pkl.gz                          <-- Quality-controlled 15-min master dataframe (28.8 MB)
│   ├── stations_13_metadata.csv                       <-- Station coordinates, elevation & land-use archetypes
│   ├── master_station_metadata.csv                    <-- Extended auditing metadata
│   ├── paper2_bidhannagar_hourly_2024_2025.csv        <-- Bidhannagar roadside ML forward transfer dataset
│   ├── paper2_182_series_distribution_ranking.csv     <-- Precomputed 182-series distribution evaluation
│   ├── paper2_distribution_reproduction.csv           <-- Distribution recomputation validation metrics
│   └── paper2_forward_transfer_results.csv            <-- Random Forest cross-year baseline results
│
├── results/
│   ├── annual_summary.csv                             <-- Table 2: 13-station conurbation criteria summaries
│   ├── paper2_photochemical_regime_indicators.csv     <-- Table 3: Station-wise photochemical indicators
│   ├── paper2_182_series_distribution_ranking.csv     <-- Table 4: 182-series MLE distribution rankings
│   └── paper2_13station_ml_transfer_and_counter_experiments.csv <-- Table 5: 13-station ML transfer & counter-experiments
│
└── scripts/
    ├── generate_paper2_figures.py                     <-- Figures 1–7 generation pipeline
    ├── compute_regime_table.py                        <-- Computes Table 3 photochemical indicators
    ├── recompute_paper2_distributions.py              <-- 182-series MLE two-parameter distribution fitting
    ├── paper2_rf_forward_transfer.py                  <-- Cross-year Random Forest transfer & counter-experiments
    ├── generate_graphical_abstract.py                 <-- Graphical abstract generator
    ├── inspect_diurnal.py                             <-- Diurnal odd-oxygen titration inspection
    └── run_proof.sh                                   <-- End-to-end execution script
```

---

## 🚀 Quick Start & Installation

```bash
git clone https://github.com/SAPTARSHI-coder/kolkata-howrah-ozone-photochemistry.git
cd kolkata-howrah-ozone-photochemistry
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Reproducing Manuscript Results & Figures

### 1. Recompute 182-Series Distribution Fitting (Table 4 & Figure 6)
Fits zero-location (`floc=0`) Lognormal, Gamma, and Weibull distributions across all 182 spatiotemporal series:
```bash
python scripts/recompute_paper2_distributions.py
```

### 2. Compute Photochemical Regime Indicators (Table 3)
Calculates station-wise $\\sum\\text{BTEX}$, $\\text{NO}_x$, $\\sum\\text{BTEX}/\\text{NO}_x$, midday $\\text{NO}_2/\\text{NO}$, and conserved $[\\text{O}_x]$:
```bash
python scripts/compute_regime_table.py
```

### 3. Run Machine Learning Forward Transfer & Chemical Counter-Experiments (Table 5 & Figure 7)
Evaluates cross-year Random Forest generalization and proves nitric oxide titration recovery:
```bash
python scripts/paper2_rf_forward_transfer.py
```

### 4. Generate All 7 Publication Figures
Generates high-resolution publication figures (300 DPI) matching the manuscript:
```bash
python scripts/generate_paper2_figures.py
```

---

## 📜 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{sadhu2026atmosphericenvironment,
  title={Ground-Level Ozone Photochemistry, Statistical Distribution Tail Modeling, and Meteorological Machine Learning Generalization across an Urban-Industrial Megacity Airshed},
  author={Sadhu, Saptarshi},
  journal={Atmospheric Environment},
  year={2026},
  publisher={Elsevier},
  note={Under Review}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
