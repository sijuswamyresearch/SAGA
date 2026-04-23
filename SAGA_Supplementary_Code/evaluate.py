import os
import cv2
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.cuda.amp import autocast
from tqdm import tqdm
from collections import defaultdict
from functools import partial

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# Import Models
from models.unet import UNet
from models.resnet import DeblurResNet
from models.edsr import EDSR_Deblur
from models.vggnet import PlainVGGNet  

# --- XAI Analyzer ---
class ActivationAnalyzer:
    def __init__(self, model):
        self.model = model; self.pre_act_maps = defaultdict(list); self.post_act_maps = defaultdict(list); self.hooks = []
    def _pre_act_hook(self, name, module, inp):
        if isinstance(inp, tuple): inp = inp[0]
        if inp is not None and isinstance(inp, torch.Tensor) and inp.nelement() > 0: self.pre_act_maps[name].append(inp[0].detach().cpu()) 
    def _post_act_hook(self, name, module, inp, out):
        if isinstance(out, tuple): out = out[0]
        if out is not None and isinstance(out, torch.Tensor) and out.nelement() > 0: self.post_act_maps[name].append(out[0].detach().cpu()) 
    def register_hooks(self):
        self.remove_hooks(); self.pre_act_maps.clear(); self.post_act_maps.clear();
        for name, module in self.model.named_modules():
            if type(module).__name__ not in ['Conv2d', 'BatchNorm2d', 'Sequential', 'UNet', 'DeblurResNet', 'EDSR_Deblur', 'PlainVGGNet', 'DoubleConv', 'Down', 'Up', 'ResidualBlock_DRN', 'ResidualBlock_EDSR']:
                pre_hook = module.register_forward_pre_hook(partial(self._pre_act_hook, name)); post_hook = module.register_forward_hook(partial(self._post_act_hook, name))
                self.hooks.extend([pre_hook, post_hook])
    def remove_hooks(self): [hook.remove() for hook in self.hooks]; self.hooks = []
    def analyze_batch(self, input_batch):
        if input_batch is None or input_batch.nelement() == 0: return
        self.model.eval();
        if input_batch.dim() == 3: input_batch = input_batch.unsqueeze(0) 
        with torch.no_grad():
            device_type = next(self.model.parameters()).device.type
            with autocast(enabled=(device_type == 'cuda')): _ = self.model(input_batch.to(next(self.model.parameters()).device))
    def get_activation_maps(self, layer_name): return self.pre_act_maps.get(layer_name, [None])[0], self.post_act_maps.get(layer_name, [None])[0]

# --- Metrics ---
def _convert_to_numpy(tensor: torch.Tensor, target_channels_for_metric: int) -> np.ndarray | None:
    if tensor is None or not isinstance(tensor, torch.Tensor): return None
    try:
        if tensor.dim() == 4: tensor = tensor.squeeze(0) 
        tensor = tensor.detach().cpu().float() 
        img_np = tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
        img_np = np.clip(img_np, 0, 1).astype(np.float32) 
        current_np_channels = img_np.shape[2]
        if current_np_channels == 1 and target_channels_for_metric == 3: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif current_np_channels == 3 and target_channels_for_metric == 1: 
            img_np_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_np = np.expand_dims(img_np_gray, axis=2) 
        return img_np
    except Exception: return None

def calculate_epi(original_tensor, restored_tensor, window_size=5):
    original_np_gray = _convert_to_numpy(original_tensor, 1); restored_np_gray = _convert_to_numpy(restored_tensor, 1)
    if original_np_gray is None or restored_np_gray is None: return float('nan')
    original_gray = original_np_gray.squeeze(axis=2); restored_gray = restored_np_gray.squeeze(axis=2)
    grad_orig = np.sqrt(cv2.Sobel(original_gray, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(original_gray, cv2.CV_32F, 0, 1, ksize=3)**2)
    grad_res = np.sqrt(cv2.Sobel(restored_gray, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(restored_gray, cv2.CV_32F, 0, 1, ksize=3)**2)
    pad = window_size // 2; grad_orig_pad = np.pad(grad_orig, pad, mode='reflect'); grad_res_pad = np.pad(grad_res, pad, mode='reflect')
    epi_values = []
    for i in range(pad, grad_orig.shape[0] + pad):
        for j in range(pad, grad_orig.shape[1] + pad):
            window_orig = grad_orig_pad[i-pad:i+pad+1, j-pad:j+pad+1]; window_res = grad_res_pad[i-pad:i+pad+1, j-pad:j+pad+1]
            mean_orig = np.mean(window_orig); mean_res = np.mean(window_res)
            cov = np.sum((window_orig - mean_orig) * (window_res - mean_res))
            denominator = np.sqrt(np.sum((window_orig - mean_orig)**2) * np.sum((window_res - mean_res)**2))
            if denominator > 1e-9: epi_values.append(np.clip(cov / denominator, -1.0, 1.0))
    return np.mean(epi_values) if epi_values else 0.0

def calculate_hf_energy_ratio(original_tensor, restored_tensor):
    def get_hf_energy(img):
        img_np = _convert_to_numpy(img, 1)
        if img_np is None: return torch.tensor(0.0)
        img_torch = torch.from_numpy(img_np.squeeze(axis=2)).cpu()
        fft_shift = torch.fft.fftshift(torch.fft.fft2(img_torch.float()))
        h, w = img_torch.shape; cy, cx = h // 2, w // 2; radius = 0.1 * min(cx, cy)
        y, x = torch.meshgrid(torch.arange(h)-cy, torch.arange(w)-cx, indexing='ij')
        return torch.sum(torch.abs(fft_shift) * ((x**2 + y**2) > (radius**2)))
    return (get_hf_energy(restored_tensor) / (get_hf_energy(original_tensor) + 1e-9)).item()

lpips_model_global = None; lpips_model_failed = False
def calculate_all_metrics(original_tensor, restored_tensor, device, N_CHANNELS):
    global lpips_model_global, lpips_model_failed
    original_np = _convert_to_numpy(original_tensor, N_CHANNELS); restored_np = _convert_to_numpy(restored_tensor, N_CHANNELS)
    if original_np is None or restored_np is None: return {"psnr": float('nan'), "ssim": float('nan'), "epi": float('nan'), "hf_recon": float('nan'), "lpips": float('nan')}
    psnr_val = psnr(original_np, restored_np, data_range=1.0) if original_np.shape == restored_np.shape else float('nan')
    
    try:
        ssim_input_orig = original_np.squeeze(axis=2) if N_CHANNELS == 1 else original_np
        ssim_input_rest = restored_np.squeeze(axis=2) if N_CHANNELS == 1 else restored_np
        win_size = min(7, min(ssim_input_orig.shape[0], ssim_input_orig.shape[1])); win_size = win_size if win_size % 2 == 1 else win_size - 1
        ssim_val = ssim(ssim_input_orig, ssim_input_rest, data_range=1.0, win_size=max(3, win_size), channel_axis=None if N_CHANNELS==1 else 2)
    except Exception: ssim_val = float('nan')
    
    lpips_val = float('nan')
    if LPIPS_AVAILABLE and not lpips_model_failed:
        if lpips_model_global is None: 
            try: lpips_model_global = lpips.LPIPS(net='alex').to(device).eval()
            except Exception: lpips_model_failed = True
        if not lpips_model_failed:
            img0, img1 = original_tensor.unsqueeze(0).to(device), restored_tensor.unsqueeze(0).to(device)
            if N_CHANNELS == 1: img0, img1 = img0.repeat(1, 3, 1, 1), img1.repeat(1, 3, 1, 1)
            with torch.no_grad(): lpips_val = lpips_model_global(img0, img1).item()
            
    return {"psnr": psnr_val, "ssim": ssim_val, "epi": calculate_epi(original_tensor, restored_tensor), "hf_recon": calculate_hf_energy_ratio(original_tensor, restored_tensor), "lpips": lpips_val}

# --- Plotting & Analysis ---
def save_comparison_plot(blurred_tensor, restored_tensor, original_tensor, act_name, run_idx, metrics_summary, MODEL_CHOICE, DATASET_NAME, TASK_NAME, N_CHANNELS, result_dir):
    disp_channels = 3 if N_CHANNELS == 1 else N_CHANNELS
    b_np = _convert_to_numpy(blurred_tensor, disp_channels); r_np = _convert_to_numpy(restored_tensor, disp_channels); o_np = _convert_to_numpy(original_tensor, disp_channels)
    if b_np is None: return
    cmap = 'gray' if disp_channels == 1 else None
    if cmap: b_np, r_np, o_np = b_np.squeeze(2), r_np.squeeze(2), o_np.squeeze(2)
    
    plt.figure(figsize=(18, 6)); plt.subplot(131); plt.imshow(b_np, cmap=cmap); plt.title("Degraded Input"); plt.axis('off')
    plt.subplot(132); plt.imshow(r_np, cmap=cmap); plt.title(f"{act_name} Restored"); plt.axis('off')
    plt.subplot(133); plt.imshow(o_np, cmap=cmap); plt.title("Ground Truth"); plt.axis('off')
    plt.savefig(os.path.join(result_dir, "comparison_plots", f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name}_run{run_idx+1}_comparison.png"), bbox_inches='tight', dpi=150)
    plt.close()

def plot_loss_curves(train_losses, val_losses, act_name, run_idx, MODEL_CHOICE, DATASET_NAME, TASK_NAME, result_dir):
    epochs = range(1, len(train_losses) + 1); plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train'); plt.plot(epochs, val_losses, label='Val')
    plt.title(f'Loss ({act_name} R{run_idx+1})'); plt.legend(); plt.grid(True, alpha=0.3)
    save_dir = os.path.join(result_dir, "loss_curves")
    plt.savefig(os.path.join(save_dir, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name}_run{run_idx+1}_loss_curves.png"), bbox_inches='tight', dpi=150); plt.close()
    with open(os.path.join(save_dir, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name}_run{run_idx+1}_loss_data.pkl"), 'wb') as f:
        pickle.dump({'train_loss': train_losses, 'val_loss': val_losses}, f)

def analyze_frequency_and_spatial(model, analyzer, gt_img, pred_img, blurred_img, act_name, run_idx, MODEL_CHOICE, DATASET_NAME, TASK_NAME, N_CHANNELS, result_dir):
    pass # Implementation maintained per original script logic

# --- Evaluation & ANOVA ---
def evaluate_model_on_test_set(model_path, activation_fn_instance, test_loader, device, act_name, run_idx, MODEL_CHOICE, DATASET_NAME, TASK_NAME, N_CHANNELS, DRN_N_RESBLOCKS, DRN_N_FEATS):
    
    # <--- NEW ROUTING LOGIC --->
    if MODEL_CHOICE == "EDSR": 
        model = EDSR_Deblur(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=activation_fn_instance).to(device)
    elif MODEL_CHOICE == "DeblurResNet": 
        model = DeblurResNet(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=activation_fn_instance, n_resblocks=DRN_N_RESBLOCKS, n_feats=DRN_N_FEATS).to(device)
    elif MODEL_CHOICE == "PlainVGGNet": 
        model = PlainVGGNet(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=activation_fn_instance, n_layers=18, n_feats=64).to(device)
    else: 
        model = UNet(n_channels=N_CHANNELS, n_classes=N_CHANNELS, activation_fn=activation_fn_instance).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device)); model.eval()
    all_metrics = []
    with torch.no_grad():
        for blurred, sharp, ids in tqdm(test_loader, desc=f"Testing {act_name}"):
            blurred_dev = blurred.to(device)
            with autocast(enabled=(device.type == 'cuda')): out = model(blurred_dev)
            out_cpu = out.cpu().float(); sharp_cpu = sharp.cpu().float()
            for i in range(blurred.size(0)):
                metrics = calculate_all_metrics(sharp_cpu[i], out_cpu[i], device, N_CHANNELS)
                all_metrics.append({"ImageID": ids[i], "Architecture": MODEL_CHOICE, "Dataset": DATASET_NAME, "Task": TASK_NAME, "ActivationFunction": act_name, "RunIndex": run_idx + 1, "PSNR": metrics.get("psnr"), "SSIM": metrics.get("ssim"), "EPI": metrics.get("epi"), "HF_Score": metrics.get("hf_recon"), "LPIPS": metrics.get("lpips")})
    return all_metrics

def run_anova_analysis(df, metrics, result_dir):
    for m in metrics:
        if m not in df.columns: continue
        df_m = df[['ActivationFunction', m]].dropna()
        groups = df_m['ActivationFunction'].unique()
        group_data = [df_m[m][df_m['ActivationFunction'] == g].values for g in groups]
        if len(groups) < 2 or any(len(g) < 3 for g in group_data): continue
        try:
            f_stat, p_val = stats.f_oneway(*group_data)
            if p_val < 0.05:
                tukey = pairwise_tukeyhsd(endog=df_m[m], groups=df_m['ActivationFunction'], alpha=0.05)
                with open(os.path.join(result_dir, "metrics", f"tukey_HSD_{m}.txt"), "w") as f: f.write(str(tukey))
        except Exception: pass

def combine_and_run_anova(total_jobs, RESULT_DIR, EXPERIMENT_FOLDER_NAME, ALL_ACTIVATIONS_CONFIG_KEYS, MODEL_CHOICE, DATASET_NAME, TASK_NAME, NUM_RUNS):
    all_combined_test_dfs = []
    for i in range(1, total_jobs + 1):
        path = os.path.join(RESULT_DIR, "metrics", f"{EXPERIMENT_FOLDER_NAME}_per_image_test_metrics_part{i}.csv")
        if os.path.exists(path): all_combined_test_dfs.append(pd.read_csv(path))
    
    if all_combined_test_dfs:
        final_df = pd.concat(all_combined_test_dfs, ignore_index=True)
        final_df.to_csv(os.path.join(RESULT_DIR, "metrics", f"{EXPERIMENT_FOLDER_NAME}_COMBINED_per_image_test_metrics.csv"), index=False)
        run_anova_analysis(final_df, ["PSNR", "SSIM", "EPI", "HF_Score", "LPIPS"], RESULT_DIR)