import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from scipy.ndimage import label as ndimage_label
from matplotlib.gridspec import GridSpec
from model.modelOld import get_model

DATASET_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"
OUTPUT_DIR = "saliency_analysis_part2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CROP_R0, CROP_C0 = 49, 24
CROP_H, CROP_W = 352, 352
VARIABLE_NAMES = [
    "Temperature",
    "Dew Point",
    "Wind Speed",
    "Wind Gust",
    "Precipitation",
    "Pressure"
]

print("=" * 80)
print("DIAGNOSING WHERE THE WEATHER COMES FROM")
print("=" * 80)

print("\n[1/5] Loading data...")
meta = torch.load(os.path.join(DATASET_DIR, "metadata.pt"), weights_only=False)
targets = torch.load(os.path.join(DATASET_DIR, "targets.pt"), weights_only=False)
times = meta["times"]
jumbo_y = meta.get("jumbo_y_idx", 225)
jumbo_x = meta.get("jumbo_x_idx", 200)

path_map = {}
for yr_dir in ["2018", "2019", "2020", "2021"]:
    p = os.path.join(DATASET_DIR, "inputs", yr_dir)
    if not os.path.isdir(p):
        continue
    for f in os.listdir(p):
        if f.endswith(".pt"):
            path_map[f] = os.path.join(p, f)

def time_to_filename(t):
    return "X_" + str(t)[:13].replace("T", "").replace("-", "").replace(":", "") + ".pt"

def load_input(i):
    fn = time_to_filename(times[i])
    return torch.load(path_map[fn], weights_only=True).float()

print("[2/5] Loading model...")
os.environ["MODEL_PATH"] = os.path.join(os.getcwd(), "best_model.pt")
os.environ["NORM_PATH"] = os.path.join(os.getcwd(), "norm_stats.pt")
model = get_model(meta)
model.eval()

SAMPLE_INDEX = 0
print(f"\n[3/5] Processing sample {SAMPLE_INDEX}...")
x_full = load_input(SAMPLE_INDEX)  # (450, 449, C)
x_crop = x_full[CROP_R0:CROP_R0+CROP_H, CROP_C0:CROP_C0+CROP_W, :] 
x_batch = x_full.unsqueeze(0).clone().detach().requires_grad_(True) 

print("[4/5] Computing per-variable gradients...")
all_saliencies = {}
all_predictions = {}

with torch.enable_grad():
    out = model(x_batch) 

    for var_idx, var_name in enumerate(VARIABLE_NAMES):
        if x_batch.grad is not None:
            x_batch.grad.zero_()
        
        output_var = out[0, var_idx]
        output_var.backward(retain_graph=True)
        grad = x_batch.grad.detach().abs()  
        saliency_full = grad.squeeze(0).sum(dim=-1)  
        saliency_crop = saliency_full[CROP_R0:CROP_R0+CROP_H, CROP_C0:CROP_C0+CROP_W]
        saliency_crop = saliency_crop / (saliency_crop.max() + 1e-8)
        saliency_smooth = torch.from_numpy(
            ndimage.gaussian_filter(saliency_crop.cpu().numpy(), sigma=2.5)
        ).float()
        saliency_smooth = saliency_smooth / (saliency_smooth.max() + 1e-8)
        
        all_saliencies[var_name] = {
            "raw": saliency_crop,
            "smooth": saliency_smooth,
            "full": saliency_full,
        }
        all_predictions[var_name] = output_var.detach().item()

input_ch0 = x_crop[..., 0].cpu().numpy() 
input_ch0_full = x_full[..., 0].cpu().numpy() 
input_ch0_norm = (input_ch0 - input_ch0.min()) / (input_ch0.max() - input_ch0.min() + 1e-8)
input_ch0_full_norm = (input_ch0_full - input_ch0_full.min()) / (input_ch0_full.max() - input_ch0_full.min() + 1e-8)

print("\nGenerating visualization 1: Per-variable saliency maps...")
fig = plt.figure(figsize=(16, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

for idx, (var_name, sal_data) in enumerate(all_saliencies.items()):
    row, col = idx // 2, idx % 2
    ax = fig.add_subplot(gs[row, col])
    
    saliency_smooth = sal_data["smooth"]
    pred_val = all_predictions[var_name]
    im = ax.imshow(input_ch0_norm, cmap="gray", alpha=0.6, origin='lower')
    im2 = ax.imshow(saliency_smooth.cpu().numpy(), cmap="hot", alpha=0.7, origin='lower')
    
    ax.set_title(f"{var_name}\nPrediction: {pred_val:.4f}", fontsize=13, weight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Sensitivity", fontsize=10)

plt.suptitle("Per-Variable Saliency Analysis: Where Does the Model Look?", 
             fontsize=16, weight="bold", y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, "01_per_variable_saliency.png"), 
            dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/01_per_variable_saliency.png")
plt.close()

print("Generating visualization 2: Full domain context...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (var_name, sal_data) in enumerate(all_saliencies.items()):
    ax = axes[idx]
    saliency_full = sal_data["full"]
    
    sal_norm = saliency_full / (saliency_full.max() + 1e-8)
    im = ax.imshow(input_ch0_full_norm, cmap="gray", alpha=0.5, origin='lower')
    im2 = ax.imshow(sal_norm.cpu().numpy(), cmap="hot", alpha=0.7, origin='lower')
    
    rect = patches.Rectangle((CROP_C0, CROP_R0), CROP_W, CROP_H,
                              linewidth=2, edgecolor='cyan', facecolor='none',
                              linestyle='--')
    ax.add_patch(rect)
    ax.plot(jumbo_x, jumbo_y, 'b*', markersize=15, markeredgecolor='white', markeredgewidth=1)
    
    ax.set_title(f"{var_name}\nFull Domain View", fontsize=12, weight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Saliency in Full Domain Context (Cyan = Model Input Window, Blue Star = Target)", 
             fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_full_domain_context.png"), 
            dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/02_full_domain_context.png")
plt.close()

print("Generating visualization 3: Comparative analysis...")
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.25)

for idx, (var_name, sal_data) in enumerate(all_saliencies.items()):
    ax_top = fig.add_subplot(gs[0, idx])
    sal_smooth = sal_data["smooth"]
    
    im = ax_top.imshow(sal_smooth.cpu().numpy(), cmap="hot", origin='lower')
    ax_top.set_title(f"{var_name}", fontsize=12, weight="bold")
    ax_top.axis("off")
    cbar = plt.colorbar(im, ax=ax_top, fraction=0.046)
    
    ax_bot = fig.add_subplot(gs[1, idx])
    sal_np = sal_smooth.cpu().numpy()
    h_profile = sal_np.mean(axis=0)
    v_profile = sal_np.mean(axis=1)
    
    ax_bot.plot(h_profile, label="Horizontal (E-W)", linewidth=2, color="red")
    ax_bot.plot(v_profile, label="Vertical (N-S)", linewidth=2, color="blue")
    ax_bot.set_title(f"{var_name} - Intensity Profile", fontsize=10)
    ax_bot.set_ylabel("Mean Sensitivity", fontsize=9)
    ax_bot.set_xlabel("Grid Position", fontsize=9)
    ax_bot.legend(fontsize=8)
    ax_bot.grid(alpha=0.3)

plt.suptitle("Saliency Maps & Spatial Intensity Profiles", fontsize=15, weight="bold")
plt.savefig(os.path.join(OUTPUT_DIR, "03_comparative_profiles.png"), 
            dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/03_comparative_profiles.png")
plt.close()

print("Generating visualization 4: Hotspot detection...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (var_name, sal_data) in enumerate(all_saliencies.items()):
    ax = axes[idx]
    sal_np = sal_data["smooth"].cpu().numpy()
    
    threshold = sal_np.max() * 0.5
    hotspot_mask = sal_np > threshold
    
    labeled_array, num_features = ndimage_label(hotspot_mask)
    im = ax.imshow(sal_np, cmap="hot", alpha=0.8, origin='lower')
    contours = ax.contour(hotspot_mask, levels=[0.5], colors='cyan', linewidths=2, origin='lower')
    
    hotspot_area = hotspot_mask.sum()
    hotspot_pct = (hotspot_area / hotspot_mask.size) * 100
    max_val = sal_np.max()
    mean_val = sal_np.mean()
    
    ax.set_title(f"{var_name}\nHotspots: {hotspot_pct:.1f}% | Max: {max_val:.3f} | Mean: {mean_val:.3f}", 
                fontsize=11, weight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("High-Sensitivity Regions (Cyan contours mark >50% of max sensitivity)", 
             fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_hotspot_detection.png"), 
            dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/04_hotspot_detection.png")
plt.close()

print("Generating visualization 5: Directional sensitivity analysis...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (var_name, sal_data) in enumerate(all_saliencies.items()):
    ax = axes[idx]
    sal_np = sal_data["smooth"].cpu().numpy()
    
    h, w = sal_np.shape
    h_mid, w_mid = h // 2, w // 2
    
    regions = {
        "N (Upstream)": sal_np[:h_mid, :].mean(),
        "S (Leeward)": sal_np[h_mid:, :].mean(),
        "W (Left)": sal_np[:, :w_mid].mean(),
        "E (Right)": sal_np[:, w_mid:].mean(),
    }
    
    # Create bar plot
    colors_dir = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
    bars = ax.bar(regions.keys(), regions.values(), color=colors_dir, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, weight='bold')
    
    ax.set_ylabel("Mean Sensitivity", fontsize=10, weight="bold")
    ax.set_title(f"{var_name}", fontsize=12, weight="bold")
    ax.set_ylim(0, max(regions.values()) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

plt.suptitle("Directional Sensitivity Analysis: Which Wind Directions Matter?", 
             fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_directional_analysis.png"), 
            dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/05_directional_analysis.png")
plt.close()

print("Generating visualization 6: Summary statistics...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

stats_data = {}
for var_name, sal_data in all_saliencies.items():
    sal_np = sal_data["smooth"].cpu().numpy()
    stats_data[var_name] = {
        "max": sal_np.max(),
        "mean": sal_np.mean(),
        "std": sal_np.std(),
        "median": np.median(sal_np),
    }

ax = axes[0, 0]
vars_list = [s for s in VARIABLE_NAMES if s in all_saliencies]
max_vals = [stats_data[v]["max"] for v in vars_list]
colors_vars = plt.cm.tab10(np.linspace(0, 1, len(vars_list)))
ax.bar(range(len(vars_list)), max_vals, color=colors_vars, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(vars_list)))
ax.set_xticklabels(vars_list, rotation=45, ha='right')
ax.set_ylabel("Maximum Sensitivity", fontsize=11, weight="bold")
ax.set_title("Peak Sensitivity by Variable", fontsize=12, weight="bold")
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
mean_vals = [stats_data[v]["mean"] for v in vars_list]
ax.bar(range(len(vars_list)), mean_vals, color=colors_vars, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(vars_list)))
ax.set_xticklabels(vars_list, rotation=45, ha='right')
ax.set_ylabel("Mean Sensitivity", fontsize=11, weight="bold")
ax.set_title("Average Sensitivity by Variable", fontsize=12, weight="bold")
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
for i, var_name in enumerate(vars_list):
    sal_np = all_saliencies[var_name]["smooth"].cpu().numpy().flatten()
    parts = ax.violinplot([sal_np], positions=[i], widths=0.7, showmeans=True)
    for pc in parts['bodies']:
        pc.set_facecolor(colors_vars[i])
        pc.set_alpha(0.7)
ax.set_xticks(range(len(vars_list)))
ax.set_xticklabels(vars_list, rotation=45, ha='right')
ax.set_ylabel("Sensitivity Value", fontsize=11, weight="bold")
ax.set_title("Sensitivity Distribution by Variable", fontsize=12, weight="bold")
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
pred_vals = [all_predictions[v] for v in vars_list]
ax.bar(range(len(vars_list)), pred_vals, color=colors_vars, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(vars_list)))
ax.set_xticklabels(vars_list, rotation=45, ha='right')
ax.set_ylabel("Predicted Value (Normalized)", fontsize=11, weight="bold")
ax.set_title("Model Predictions for Sample", fontsize=12, weight="bold")
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_summary_statistics.png"), 
            dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/06_summary_statistics.png")
plt.close()

print("Generating visualization 7: Compass-oriented heatmaps...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (var_name, sal_data) in enumerate(all_saliencies.items()):
    ax = axes[idx]
    sal_np = sal_data["smooth"].cpu().numpy()
    
    im = ax.imshow(sal_np, cmap="hot", origin='lower', aspect='auto')
    
    h, w = sal_np.shape
    arrow_props = dict(head_width=15, head_length=15, fc='white', ec='white', linewidth=2)
    
    ax.arrow(w/2, h*0.95, 0, h*0.08, **arrow_props)
    ax.text(w/2, h*1.06, 'N', fontsize=16, weight='bold', color='white', 
            ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.arrow(w/2, h*0.05, 0, -h*0.08, **arrow_props)
    ax.text(w/2, h*-0.06, 'S', fontsize=16, weight='bold', color='white', 
            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.arrow(w*0.95, h/2, w*0.08, 0, **arrow_props)
    ax.text(w*1.06, h/2, 'E', fontsize=16, weight='bold', color='white', 
            ha='left', va='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.arrow(w*0.05, h/2, -w*0.08, 0, **arrow_props)
    ax.text(w*-0.06, h/2, 'W', fontsize=16, weight='bold', color='white', 
            ha='right', va='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title(f"{var_name}\nSensitivity Map", fontsize=13, weight="bold")
    ax.set_xlim(-w*0.15, w*1.15)
    ax.set_ylim(-h*0.15, h*1.15)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Sensitivity", fontsize=10)

plt.suptitle("Compass-Oriented Saliency Maps: Geographic Sensitivity Distribution\n(North at top, South at bottom, East at right, West at left)", 
             fontsize=15, weight="bold", y=0.98)
plt.savefig(os.path.join(OUTPUT_DIR, "07_compass_oriented_heatmaps.png"), 
            dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/07_compass_oriented_heatmaps.png")
plt.close()

print("\nSaving tensor data...")
for var_name, sal_data in all_saliencies.items():
    filename = var_name.lower().replace(" ", "_")
    torch.save(sal_data["smooth"], os.path.join(OUTPUT_DIR, f"saliency_{filename}.pt"))
    torch.save(sal_data["raw"], os.path.join(OUTPUT_DIR, f"saliency_raw_{filename}.pt"))

torch.save(x_crop, os.path.join(OUTPUT_DIR, "input_crop.pt"))
torch.save(torch.tensor(list(all_predictions.values())), os.path.join(OUTPUT_DIR, "predictions.pt"))