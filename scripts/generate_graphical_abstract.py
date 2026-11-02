"""
Generate Publication-Grade Graphical Abstract for Atmospheric Environment (Elsevier)
Image dimensions: 1600 x 850 pixels at 300 DPI (approx 2:1 aspect ratio)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Main Title Banner
ax.text(50, 96.5, "Ground-Level Ozone Photochemistry, Tail Modeling & ML Transferability across Kolkata–Howrah",
        fontsize=15, fontweight='bold', ha='center', va='center', color='#1a252f')
ax.text(50, 93.0, "13 CAAQMS Stations | 2024–2025 Continuous Telemetry (16.1M Records) | Elsevier Atmospheric Environment",
        fontsize=10.5, fontstyle='italic', ha='center', va='center', color='#7f8c8d')

# Panel 1: (x: 2 to 32, y: 5 to 88) - Statistical Distribution Modeling
# Panel 2: (x: 34 to 66, y: 5 to 88) - Diurnal Chemistry & Odd-Oxygen
# Panel 3: (x: 68 to 98, y: 5 to 88) - Machine Learning & Counter-Experiment

# --- PANEL 1: Statistical Tail Modeling ---
p1_bg = patches.FancyBboxPatch((2.5, 6), 30, 82, boxstyle="round,pad=0.5",
                              ec="#2980b9", fc="#ebf5fb", lw=1.8)
ax.add_patch(p1_bg)

# Panel 1 Header
ax.text(17.5, 84.5, "1. Statistical Tail Modeling (182 Series)",
        fontsize=12, fontweight='bold', ha='center', va='center', color='#1a5276')
ax.text(17.5, 81.5, "Testing Kolmogorov/Ott Multiplicative Dilution vs Weibull",
        fontsize=8.5, ha='center', va='center', color='#566573')

# Donut / Bar chart for distribution win rates
categories = ['Lognormal', 'Gamma', 'Weibull']
win_rates = [81.3, 15.4, 3.3]
bar_colors = ['#27ae60', '#f39c12', '#c0392b']

for i, (cat, val, col) in enumerate(zip(categories, win_rates, bar_colors)):
    y_pos = 71 - i * 11
    # Label
    ax.text(5, y_pos + 3.5, f"{cat} Distribution", fontsize=10, fontweight='bold', color='#2c3e50')
    ax.text(29, y_pos + 3.5, f"{val}%", fontsize=10.5, fontweight='bold', color=col, ha='right')
    # Bar background
    bar_bg = patches.Rectangle((5, y_pos), 25, 2.2, color='#d5dbdb')
    ax.add_patch(bar_bg)
    # Bar fill
    bar_fill = patches.Rectangle((5, y_pos), 25 * (val / 100), 2.2, color=col)
    ax.add_patch(bar_fill)

# Key takeaway box for Panel 1
box1 = patches.FancyBboxPatch((4.5, 9.5), 26, 25, boxstyle="round,pad=0.3",
                             ec="#aed6f1", fc="#ffffff", lw=1.2)
ax.add_patch(box1)
ax.text(17.5, 31, "Physical Dispersion Mechanisms", fontsize=9.5, fontweight='bold', ha='center', color='#1b4f72')
ax.text(6, 26, "• Lognormal Optimality (81.3% AIC, 74.7% KS):\n  100% VOCs, 73.1% O3 confirm successive\n  multiplicative turbulent dilution in boundary layer.",
        fontsize=8, color='#2c3e50', va='top')
ax.text(6, 17, "• Gamma Role (15.4%):\n  Optimal in 26.9% PM2.5 & CO (discrete Poisson\n  combustion emissions competing with dilution).",
        fontsize=8, color='#2c3e50', va='top')
ax.text(6, 11.5, "• Universal Weibull Model Decisively Rejected:\n  Optimal in only 3.3% (Mean ΔAIC = +1,264.4).",
        fontsize=8, fontweight='bold', color='#c0392b', va='top')


# --- PANEL 2: Diurnal Photochemistry & NO Titration ---
p2_bg = patches.FancyBboxPatch((35, 6), 30, 82, boxstyle="round,pad=0.5",
                              ec="#e67e22", fc="#fef9e7", lw=1.8)
ax.add_patch(p2_bg)

ax.text(50, 84.5, "2. Photochemistry & NO Titration",
        fontsize=12, fontweight='bold', ha='center', va='center', color='#b9770e')
ax.text(50, 81.5, "Odd-Oxygen Partitioning: [Ox] = [O3] + [NO2]",
        fontsize=8.5, ha='center', va='center', color='#566573')

# Micro-environment comparison cards
# Suburban Card
sub_card = patches.FancyBboxPatch((37, 57), 26, 20.5, boxstyle="round,pad=0.3",
                                 ec="#27ae60", fc="#ffffff", lw=1.2)
ax.add_patch(sub_card)
ax.text(50, 74.5, "Suburban / Parkland (SVSPA & VC)", fontsize=9.5, fontweight='bold', ha='center', color='#1e8449')
ax.text(38.5, 70, "• Unimodal solar-driven curve:\n  Midday O3 peak: 41.2–52.5 µg/m³ at 13–14 IST\n  Afternoon NO trough: 4.6–12.1 µg/m³\n• Driven by actinic flux and thermal mixing.",
        fontsize=8, color='#2c3e50', va='top')

# Roadside Card
rd_card = patches.FancyBboxPatch((37, 33), 26, 21.5, boxstyle="round,pad=0.3",
                                ec="#e74c3c", fc="#ffffff", lw=1.2)
ax.add_patch(rd_card)
ax.text(50, 51.5, "Urban Roadside (Bidhannagar & RBU)", fontsize=9.5, fontweight='bold', ha='center', color='#922b21')
ax.text(38.5, 47, "• Rush-hour vehicular titration sink:\n  NO + O3 → NO2 + O2\n  Suppresses nocturnal/rush-hour O3 to <5 µg/m³\n• Odd-Oxygen Reservoir ([Ox]) is Conserved:\n  [Ox] = 45.9–65.2 µg/m³ confirms persistent\n  oxidizing capacity partitioned into NO2.",
        fontsize=8, color='#2c3e50', va='top')

# Regime indicator summary
box2 = patches.FancyBboxPatch((37, 9.5), 26, 20.5, boxstyle="round,pad=0.3",
                             ec="#f39c12", fc="#ffffff", lw=1.2)
ax.add_patch(box2)
ax.text(50, 27, "Empirical Precursor Ratios (Screening)", fontsize=9.5, fontweight='bold', ha='center', color='#b9770e')
ax.text(38.5, 22.5, "• ΣBTEX / NOx < 0.45: VOC-sensitive proxy\n  (Curbside & traffic: FW 0.25, VC 0.35, BN 0.40)\n• Midday NO2/NO > 2.5: Photochemically aged\n  (BOG 5.71, DN 3.91, BG 2.88, SVSPA 2.55)\n• Screening metric: unmeasured alkenes & isoprene\n  dominate true ambient OH reactivity (kOH[VOC]).",
        fontsize=7.8, color='#2c3e50', va='top')


# --- PANEL 3: Machine Learning & Counter-Experiment ---
p3_bg = patches.FancyBboxPatch((67.5, 6), 30, 82, boxstyle="round,pad=0.5",
                              ec="#27ae60", fc="#eafaf1", lw=1.8)
ax.add_patch(p3_bg)

ax.text(82.5, 84.5, "3. Cross-Year ML & Verification",
        fontsize=12, fontweight='bold', ha='center', va='center', color='#1e8449')
ax.text(82.5, 81.5, "Trained on 2024 Records → Tested on Unseen 2025 Data",
        fontsize=8.5, ha='center', va='center', color='#566573')

# Contrast Bar Chart: Suburban vs Roadside
ax.text(82.5, 75.5, "Forward Meteorological Transfer (R²)", fontsize=10, fontweight='bold', ha='center', color='#145a32')

# Suburban
ax.text(70, 71, "Barrackpore (Suburban):", fontsize=8.5, fontweight='bold', color='#2c3e50')
ax.text(94, 71, "R² = +0.68", fontsize=9.5, fontweight='bold', color='#27ae60', ha='right')
b1 = patches.Rectangle((70, 68), 24 * (0.68 / 1.0), 1.8, color='#27ae60')
ax.add_patch(b1)

# Victoria Memorial
ax.text(70, 64, "Victoria Memorial (Park):", fontsize=8.5, fontweight='bold', color='#2c3e50')
ax.text(94, 64, "R² = +0.59", fontsize=9.5, fontweight='bold', color='#27ae60', ha='right')
b2 = patches.Rectangle((70, 61), 24 * (0.59 / 1.0), 1.8, color='#2ecc71')
ax.add_patch(b2)

# Roadside Breakdown
ax.text(70, 57, "Bidhannagar (Roadside):", fontsize=8.5, fontweight='bold', color='#2c3e50')
ax.text(94, 57, "R² = -0.26", fontsize=9.5, fontweight='bold', color='#c0392b', ha='right')
b3 = patches.Rectangle((70, 54), 10, 1.8, color='#c0392b')
ax.add_patch(b3)

# Counter-Experiment Box (Mathematical Proof)
box3 = patches.FancyBboxPatch((69.5, 9.5), 26, 40, boxstyle="round,pad=0.3",
                             ec="#27ae60", fc="#ffffff", lw=1.5)
ax.add_patch(box3)
ax.text(82.5, 46.5, "Diagnostic Counter-Experiment", fontsize=10, fontweight='bold', ha='center', color='#145a32')
ax.text(82.5, 43.5, "(Proving NO Titration as Root Cause)", fontsize=8, fontstyle='italic', ha='center', color='#7f8c8d')

ax.text(71, 39, "1. Met-Only Roadside Model:\n   R² = -0.2670  (Negative Transfer)", fontsize=8, color='#c0392b')
ax.text(71, 32.5, "2. Adding Co-Located Chemical Sinks:\n   Met + NO + NO2 → R² = -0.0764 (Gain +0.19)\n   Met + Reactive Suite → R² = +0.0195", fontsize=8, color='#2980b9')
ax.text(71, 23.5, "3. Cross-Validation Chemical Accuracy:\n   5-Fold CV Met-Only: R² = 0.6808\n   5-Fold CV + NO & NO2: R² = 0.7800", fontsize=8, color='#1e8449')
ax.text(71, 14.5, "4. Odd-Oxygen Photochemical Closure:\n   Modeling [Ox] = [O3] + [NO2] yields:\n   R² = 0.9533  (Near-Perfect Closure!)", fontsize=8, fontweight='bold', color='#1e8449')

output_path1 = "/Users/saptarshisadhu/Desktop/5_PAPERS/02_Paper_2_Atmospheric_Environment/figures/graphical_abstract.png"
output_path2 = "/Users/saptarshisadhu/Desktop/5_PAPERS/02_Paper_2_Atmospheric_Environment/03_SUBMISSION_DOCUMENTS/graphical_abstract.png"

plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"--> Saved {output_path1}")
print(f"--> Saved {output_path2}")
