"""
Paper 2: Recomputation and Ranking of Statistical Distributions (Weibull, Gamma, Lognormal)
Fits 2-parameter distributions (floc=0) across 182 pollutant time-series across all 13 CAAQMS stations.
Calculates log-likelihood, AIC, BIC, and Kolmogorov-Smirnov test statistics.
"""
import os
import sys
import pickle
import gzip
import numpy as np
import pandas as pd
import scipy.stats as stats

script_dir = os.path.dirname(os.path.abspath(__file__))
ranking_csv = os.path.join(script_dir, 'paper2_182_series_distribution_ranking.csv')
out_csv = os.path.join(script_dir, 'paper2_distribution_reproduction.csv')

cache_candidates = [
    os.environ.get('DATA_CACHE', ''),
    os.path.abspath(os.path.join(script_dir, '../../../../Research_Recovery_Audit/phase2/data_cache/master_cleaned.pkl.gz')),
    '/Users/saptarshisadhu/Desktop/Research_Recovery_Audit/phase2/data_cache/master_cleaned.pkl.gz'
]
cache_pkl = next((c for c in cache_candidates if c and os.path.exists(c)), None)

if cache_pkl and os.path.exists(cache_pkl):
    print(f"Loading master cleaned telemetry cache from: {cache_pkl}")
    with gzip.open(cache_pkl, 'rb') as f:
        df_master = pickle.load(f)

    pollutants = ['PM2.5', 'PM10', 'Ozone', 'NOx', 'SO2', 'CO', 'VOC_Total']
    stations = df_master['Station'].unique()
    years = [2024, 2025]
    records = []

    for pol in pollutants:
        for st in stations:
            for yr in years:
                sub = df_master[(df_master['Station'] == st) & (df_master['Year'] == yr)]
                if pol not in sub.columns:
                    continue
                
                raw_series = sub[pol].dropna()
                n_raw = len(raw_series)
                
                pos_series = raw_series[raw_series > 0]
                n_pos = len(pos_series)
                n_dropped = n_raw - n_pos
                
                if n_pos < 50:
                    continue
                    
                x = pos_series.values
                
                # 1. Fit Weibull: 2 parameters (shape, scale), fix loc=0
                try:
                    wb_shape, wb_loc, wb_scale = stats.weibull_min.fit(x, floc=0)
                    ll_wb = np.sum(stats.weibull_min.logpdf(x, wb_shape, loc=0, scale=wb_scale))
                    aic_wb = 2 * 2 - 2 * ll_wb
                    bic_wb = 2 * np.log(n_pos) - 2 * ll_wb
                    ks_wb, _ = stats.kstest(x, 'weibull_min', args=(wb_shape, 0, wb_scale))
                except Exception:
                    wb_shape, wb_scale, ll_wb, aic_wb, bic_wb, ks_wb = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    
                # 2. Fit Gamma: 2 parameters (shape, scale), fix loc=0
                try:
                    gm_shape, gm_loc, gm_scale = stats.gamma.fit(x, floc=0)
                    ll_gm = np.sum(stats.gamma.logpdf(x, gm_shape, loc=0, scale=gm_scale))
                    aic_gm = 2 * 2 - 2 * ll_gm
                    bic_gm = 2 * np.log(n_pos) - 2 * ll_gm
                    ks_gm, _ = stats.kstest(x, 'gamma', args=(gm_shape, 0, gm_scale))
                except Exception:
                    gm_shape, gm_scale, ll_gm, aic_gm, bic_gm, ks_gm = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    
                # 3. Fit Lognormal: 2 parameters (s=sigma, scale=exp(mu)), fix loc=0
                try:
                    ln_s, ln_loc, ln_scale = stats.lognorm.fit(x, floc=0)
                    ll_ln = np.sum(stats.lognorm.logpdf(x, ln_s, loc=0, scale=ln_scale))
                    aic_ln = 2 * 2 - 2 * ll_ln
                    bic_ln = 2 * np.log(n_pos) - 2 * ll_ln
                    ks_ln, _ = stats.kstest(x, 'lognorm', args=(ln_s, 0, ln_scale))
                except Exception:
                    ln_s, ln_scale, ll_ln, aic_ln, bic_ln, ks_ln = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

                aics = {'Weibull': aic_wb, 'Gamma': aic_gm, 'Lognormal': aic_ln}
                valid_aics = {k: v for k, v in aics.items() if not np.isnan(v)}
                best_dist_aic = min(valid_aics, key=valid_aics.get) if valid_aics else 'None'
                
                kss = {'Weibull': ks_wb, 'Gamma': ks_gm, 'Lognormal': ks_ln}
                valid_kss = {k: v for k, v in kss.items() if not np.isnan(v)}
                best_dist_ks = min(valid_kss, key=valid_kss.get) if valid_kss else 'None'
                
                is_weibull_best = (best_dist_aic == 'Weibull')
                
                records.append({
                    'Pollutant': pol,
                    'Station': st,
                    'Year': yr,
                    'N_Raw': n_raw,
                    'N_Positive_Fitted': n_pos,
                    'N_Zero_Or_Negative_Dropped': n_dropped,
                    'Weibull_Shape_k': round(wb_shape, 4) if not np.isnan(wb_shape) else np.nan,
                    'Weibull_Scale_lambda': round(wb_scale, 4) if not np.isnan(wb_scale) else np.nan,
                    'Weibull_AIC': round(aic_wb, 2) if not np.isnan(aic_wb) else np.nan,
                    'Weibull_BIC': round(bic_wb, 2) if not np.isnan(bic_wb) else np.nan,
                    'Weibull_KS_Dn': round(ks_wb, 4) if not np.isnan(ks_wb) else np.nan,
                    'Gamma_Shape_alpha': round(gm_shape, 4) if not np.isnan(gm_shape) else np.nan,
                    'Gamma_Scale_beta': round(gm_scale, 4) if not np.isnan(gm_scale) else np.nan,
                    'Gamma_AIC': round(aic_gm, 2) if not np.isnan(aic_gm) else np.nan,
                    'Gamma_BIC': round(bic_gm, 2) if not np.isnan(bic_gm) else np.nan,
                    'Gamma_KS_Dn': round(ks_gm, 4) if not np.isnan(ks_gm) else np.nan,
                    'Lognormal_Sigma_s': round(ln_s, 4) if not np.isnan(ln_s) else np.nan,
                    'Lognormal_Scale': round(ln_scale, 4) if not np.isnan(ln_scale) else np.nan,
                    'Lognormal_AIC': round(aic_ln, 2) if not np.isnan(aic_ln) else np.nan,
                    'Lognormal_BIC': round(bic_ln, 2) if not np.isnan(bic_ln) else np.nan,
                    'Lognormal_KS_Dn': round(ks_ln, 4) if not np.isnan(ks_ln) else np.nan,
                    'Best_Distribution_by_AIC': best_dist_aic,
                    'Best_Distribution_by_KS': best_dist_ks,
                    'Original_Manuscript_Claim': 'Weibull Universal',
                    'Independent_Challenge_Verdict': 'CONFIRMED WEIBULL' if is_weibull_best else 'CONTRADICTED (NON-WEIBULL SUPERIOR)',
                    'Delta_AIC (Weibull - Best)': round(aic_wb - valid_aics[best_dist_aic], 2) if valid_aics else np.nan
                })

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False)
    print(f"Successfully generated {out_csv} with {len(records)} fitted distribution records.")
elif os.path.exists(ranking_csv):
    print(f"Loading precomputed 182-series distribution rankings from: {ranking_csv}")
    df_out = pd.read_csv(ranking_csv)
else:
    print(f"ERROR: Neither data cache nor ranking CSV found in {script_dir}")
    sys.exit(1)

print("\n=== SUMMARY OF BEST FIT BY AIC ACROSS ALL 182 SERIES ===")
summary_aic = pd.crosstab(df_out['Pollutant'], df_out['Best_Distribution_by_AIC'], margins=True)
print(summary_aic)

print("\n=== SUMMARY OF BEST FIT BY KS STATISTIC ACROSS ALL 182 SERIES ===")
summary_ks = pd.crosstab(df_out['Pollutant'], df_out['Best_Distribution_by_KS'], margins=True)
print(summary_ks)
