import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy import ndimage
from model.modelOld import get_model

DATASET_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"
SAMPLE_INDEX = 0
OUTPUT_DIR = "saliency_outputs_news"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CROP_R0, CROP_C0 = 49, 24
CROP_H, CROP_W = 352, 352

meta = torch.load(os.path.join(DATASET_DIR, "metadata.pt"), weights_only=False)
targets = torch.load(os.path.join(DATASET_DIR, "targets.pt"), weights_only=False)
times = meta["times"]
jumbo_y = meta.get("jumbo_y_idx", 225)
jumbo_x = meta.get("jumbo_x_idx", 200)

path_map = {}
for yr_dir in ["2018", "2019", "2020", "2021"]:
    p = os.path.join(DATASET_DIR, "inputs", yr_dir)
    if not os.path.isdir(p): continue
    for f in os.listdir(p):
        if f.endswith(".pt"): 
            path_map[f] = os.path.join(p, f)

def time_to_filename(t):
    return "X_" + str(t)[:13].replace("T", "").replace("-", "").replace(":", "") + ".pt"

def load_input(i):
    return torch.load(path_map[time_to_filename(times[i])], weights_only=True).float()

os.environ["MODEL_PATH"] = os.path.join(os.getcwd(), "best_model.pt")
os.environ["NORM_PATH"] = os.path.join(os.getcwd(), "norm_stats.pt")
model = get_model(meta)
model.eval()

x_full = load_input(SAMPLE_INDEX)  
x_crop = x_full[CROP_R0:CROP_R0+CROP_H, CROP_C0:CROP_C0+CROP_W, :]  
x_batch = x_full.unsqueeze(0).clone().detach().requires_grad_(True)  

with torch.enable_grad():
    out = model(x_batch)
    target_output = out[0, 0]  
    target_output.backward()

grad = x_batch.grad.detach().abs() 
saliency_full = grad.squeeze(0).sum(dim=-1) 
saliency_crop = saliency_full[CROP_R0:CROP_R0+CROP_H, CROP_C0:CROP_C0+CROP_W]  
saliency_crop = saliency_crop / (saliency_crop.max() + 1e-8)

saliency_smooth = torch.from_numpy(
    ndimage.gaussian_filter(saliency_crop.cpu().numpy(), sigma=3)
).float()

saliency_smooth = saliency_smooth / (saliency_smooth.max() + 1e-8)

input_ch0 = x_crop[..., 0].cpu().numpy() 
input_ch0_full = x_full[..., 0].cpu().numpy() 

input_ch0_norm = (input_ch0 - input_ch0.min()) / (input_ch0.max() - input_ch0.min() + 1e-8)
input_ch0_full_norm = (input_ch0_full - input_ch0_full.min()) / (input_ch0_full.max() - input_ch0_full.min() + 1e-8)

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes[0, 0].imshow(input_ch0_full_norm, cmap="viridis", origin='lower')
axes[0, 0].set_title("Full Domain - Input Channel 0\n(with crop window and target)", fontsize=12, weight="bold")
rect = patches.Rectangle((CROP_C0, CROP_R0), CROP_W, CROP_H, 
                          linewidth=2, edgecolor='red', facecolor='none', 
                          linestyle='--', label='Model crop window')
axes[0, 0].add_patch(rect)

axes[0, 0].plot(jumbo_x, jumbo_y, 'r*', markersize=20, label='Target (Jumbo)', markeredgecolor='white', markeredgewidth=1.5)
axes[0, 0].legend(loc='upper right')
axes[0, 0].axis("off")

axes[0, 1].imshow(input_ch0, cmap="viridis", origin='lower')
axes[0, 1].set_title("Input to Model (Cropped)\n352×352 region", fontsize=12, weight="bold")
axes[0, 1].axis("off")
cbar0 = plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
cbar0.set_label("Value")

im1 = axes[1, 0].imshow(saliency_smooth.cpu().numpy(), cmap="hot", origin='lower')
axes[1, 0].set_title("Saliency Map (Smoothed)\nWhere does model look?", fontsize=12, weight="bold")
axes[1, 0].axis("off")
cbar1 = plt.colorbar(im1, ax=axes[1, 0])
cbar1.set_label("Gradient Magnitude")

axes[1, 1].imshow(input_ch0_norm, cmap="gray", alpha=0.7, origin='lower')
im2 = axes[1, 1].imshow(saliency_smooth.cpu().numpy(), cmap="hot", alpha=0.6, origin='lower')
axes[1, 1].set_title("Heatmap Overlay\n(Input regions → Output sensitivity)", fontsize=12, weight="bold")
axes[1, 1].axis("off")
cbar2 = plt.colorbar(im2, ax=axes[1, 1])
cbar2.set_label("Sensitivity")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"saliency_analysis_{SAMPLE_INDEX}.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/saliency_analysis_{SAMPLE_INDEX}.png")

torch.save(x_crop, os.path.join(OUTPUT_DIR, f"input_crop_{SAMPLE_INDEX}.pt"))
torch.save(torch.from_numpy(saliency_smooth.cpu().numpy()), os.path.join(OUTPUT_DIR, f"saliency_{SAMPLE_INDEX}.pt"))
torch.save(out.detach().cpu(), os.path.join(OUTPUT_DIR, f"prediction_{SAMPLE_INDEX}.pt"))

print(f"Prediction: {out.detach().cpu().numpy()[0]}")
print(f"Saliency range: [{saliency_smooth.min():.3f}, {saliency_smooth.max():.3f}]")
print(f"Target location (Jumbo): row={jumbo_y}, col={jumbo_x}")
print(f"Crop window: rows [{CROP_R0}, {CROP_R0+CROP_H}], cols [{CROP_C0}, {CROP_C0+CROP_W}]")
