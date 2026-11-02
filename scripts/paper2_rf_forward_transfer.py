"""
Paper 2: Atmospheric Environment
Cross-Year Machine Learning Forward Transfer Reproducibility Script
Target: Surface Ozone Generalization at Bidhannagar (BN) Roadside Corridor
Evaluates forward transfer: Model trained on 2024 hourly telemetry, evaluated on unseen 2025 data.
Reproduces R^2 ≈ -0.2579 (and RMSE ≈ 13.91, MAE ≈ 10.26).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def run_forward_transfer():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Locate data: check bundled CSV first, then workspace fallback
    csv_bundled = os.path.join(script_dir, "paper2_bidhannagar_hourly_2024_2025.csv")
    if os.path.exists(csv_bundled):
        df = pd.read_csv(csv_bundled)
        h24 = df[df['Year'] == 2024].copy()
        h25 = df[df['Year'] == 2025].copy()
    else:
        # Fallback to master cleaned Excel files
        data_candidates = [
            os.path.abspath(os.path.join(script_dir, "../../../02_PROCESSED_DATA/Cleaned_Station_Data")),
            "/Users/saptarshisadhu/Desktop/papers/02_PROCESSED_DATA/Cleaned_Station_Data"
        ]
        base_dir = next((d for d in data_candidates if os.path.exists(d)), None)
        if base_dir is None:
            raise FileNotFoundError("Could not find Bidhannagar telemetry data files.")
            
        f24 = os.path.join(base_dir, "BN_2024_MLL.xlsx")
        f25 = os.path.join(base_dir, "BN_2025_MLL.xlsx")
        
        df24 = pd.read_excel(f24)
        df25 = pd.read_excel(f25)
        df24['DateTime'] = pd.to_datetime(df24['DateTime'])
        df25['DateTime'] = pd.to_datetime(df25['DateTime'])
        
        h24 = df24.set_index('DateTime').resample('1h').mean(numeric_only=True).dropna(subset=['Ozone']).reset_index()
        h25 = df25.set_index('DateTime').resample('1h').mean(numeric_only=True).dropna(subset=['Ozone']).reset_index()
        h24['Hour'] = h24['DateTime'].dt.hour
        h25['Hour'] = h25['DateTime'].dt.hour

    features = ['AT', 'RH', 'WS', 'WD', 'BP', 'RF', 'Hour']
    target = 'Ozone'
    
    tr = h24[features + [target]].dropna()
    te = h25[features + [target]].dropna()
    
    # Random Forest Regressor using calibrated hyper-parameters matching Section 2.4
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=4,
        random_state=123,
        n_jobs=-1
    )
    rf.fit(tr[features], tr[target])
    
    preds = rf.predict(te[features])
    
    r2 = r2_score(te[target], preds)
    rmse = np.sqrt(mean_squared_error(te[target], preds))
    mae = mean_absolute_error(te[target], preds)
    
    out_df = pd.DataFrame([{
        'Station': 'Bidhannagar',
        'Model': 'Random Forest',
        'Train_Year': 2024,
        'Test_Year': 2025,
        'R2': round(r2, 4),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2)
    }])
    
    out_csv = os.path.join(script_dir, "paper2_forward_transfer_results.csv")
    out_df.to_csv(out_csv, index=False)
    
    print("=" * 60)
    print("PAPER 2: RANDOM FOREST FORWARD TRANSFER RESULTS (2024 -> 2025)")
    print(f"Station: Bidhannagar (BN) Roadside Corridor")
    print(f"Test R^2:  {r2:.4f} (Manuscript reported: -0.2579)")
    print(f"Test RMSE: {rmse:.2f} ug/m3 (Manuscript reported: 13.91)")
    print(f"Test MAE:  {mae:.2f} ug/m3 (Manuscript reported: 10.26)")
    print(f"Results saved to: {out_csv}")
    print("=" * 60)
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

if __name__ == "__main__":
    run_forward_transfer()
