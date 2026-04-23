# -*- coding: utf-8 -*-
"""
COMPLETE LRP ANALYSIS WITH STATISTICAL SIGNIFICANCE (t-test & Cohen's d_z)
Healthcare Analytics - SAGA Supplementary Code

Final Features Included:
1. True LRP (Montavon et al.): alpha1-beta0 rule.
2. Hook Filtering: Ignores internal custom AF components.
3. Robust Normalization: Fair percentile clipping across ALL models.
4. Bhati et al. Algorithm 1: Dynamic neuron selection for ReLU, FReLU, and SAGA.
5. Sparsity & Validation: MSE and SMAPE reconstruction validation.
6. Statistical Significance: Direct paired t-tests and Cohen's d_z estimation.
"""

import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy import stats
import gc
import warnings
from collections import OrderedDict

# Import architectures directly from our modular codebase
from models.resnet import DeblurResNet
from models.saga_layer import FReLU, SAGA

warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Configuration
N_CHANNELS = 1
TARGET_SIZE = (256, 256)
DRN_N_RESBLOCKS = 16
DRN_N_FEATS = 64

# ============================================================================
# 1. DATASET CLASS & HELPER FUNCTIONS
# ============================================================================
class MedicalImageRestorationDataset:
    def __init__(self, root_dir, split='train', task_folder='data_dblur', degraded_folder='blur', transform=None, target_size=None, input_channels=1):
        import glob
        self.task_split_path = os.path.join(root_dir, task_folder, split)
        self.sharp_dir = os.path.join(self.task_split_path, 'sharp')
        self.degraded_dir = os.path.join(self.task_split_path, degraded_folder)
        self.transform = transform
        self.target_size = target_size
        self.input_channels = input_channels
        self.sharp_files = []
        for ext in ['*.pt', '*.pth']:
            self.sharp_files.extend(sorted(glob.glob(os.path.join(self.sharp_dir, ext))))
        self.image_pairs = []
        self.image_ids = []
        for sharp_path in self.sharp_files:
            filename = os.path.basename(sharp_path)
            degraded_path = os.path.join(self.degraded_dir, filename)
            if os.path.isfile(degraded_path):
                self.image_pairs.append((sharp_path, degraded_path))
                self.image_ids.append(filename)

    def __len__(self): return len(self.image_pairs)

    def __getitem__(self, idx):
        sharp_path, degraded_path = self.image_pairs[idx]
        image_id = self.image_ids[idx]
        sharp_tensor = torch.load(sharp_path, weights_only=False).float()
        degraded_tensor = torch.load(degraded_path, weights_only=False).float()

        if sharp_tensor.dim() == 3 and sharp_tensor.shape[2] == self.input_channels: sharp_tensor = sharp_tensor.permute(2, 0, 1)
        elif sharp_tensor.dim() == 2 and self.input_channels == 1: sharp_tensor = sharp_tensor.unsqueeze(0)
        
        if degraded_tensor.dim() == 3 and degraded_tensor.shape[2] == self.input_channels: degraded_tensor = degraded_tensor.permute(2, 0, 1)
        elif degraded_tensor.dim() == 2 and self.input_channels == 1: degraded_tensor = degraded_tensor.unsqueeze(0)

        sharp_tensor = torch.clamp(sharp_tensor, 0.0, 1.0)
        degraded_tensor = torch.clamp(degraded_tensor, 0.0, 1.0)

        if self.target_size:
            resize = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            sharp_tensor = resize(sharp_tensor)
            degraded_tensor = resize(degraded_tensor)
        if self.transform:
            degraded_tensor = self.transform(degraded_tensor)
            sharp_tensor = self.transform(sharp_tensor)

        return degraded_tensor, sharp_tensor, image_id

def inject_lrp_hooks(model):
    model.activations = OrderedDict()
    model.hook_handles = []
    
    def forward_hook(module, inp, out):
        model.activations[module] = inp[0].detach()
        
    modules_to_hook = [model.head]
    for block in model.body:
        for m in block.body:
            if isinstance(m, nn.Conv2d):
                modules_to_hook.append(m)
    modules_to_hook.append(model.tail)

    for module in modules_to_hook:
        model.hook_handles.append(module.register_forward_hook(forward_hook))

def remove_lrp_hooks(model):
    for handle in model.hook_handles: handle.remove()
    model.hook_handles = []

# ============================================================================
# 2. TRUE LRP & NEURON SELECTION
# ============================================================================
class TrueLRP_Analyzer:
    def __init__(self, model, device='cuda', epsilon=1e-6):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.model.eval()

    def lrp_alpha1_beta0(self, layer, activation, relevance_next):
        w_plus = torch.clamp(layer.weight, min=0.0)
        
        z_k = F.conv2d(activation, w_plus, bias=None, 
                       stride=layer.stride, padding=layer.padding, groups=layer.groups) + self.epsilon
        
        if relevance_next.shape[2:] != z_k.shape[2:]:
            relevance_next = F.interpolate(relevance_next, size=z_k.shape[2:], mode='bilinear', align_corners=False)
            
        if relevance_next.shape[1] != z_k.shape[1]:
            if relevance_next.shape[1] == 1:
                relevance_next = relevance_next.expand(-1, z_k.shape[1], -1, -1)
            else:
                relevance_next = torch.sum(relevance_next, dim=1, keepdim=True).expand(-1, z_k.shape[1], -1, -1)

        s_k = relevance_next / z_k
        c_j = F.conv_transpose2d(s_k, w_plus, 
                                 stride=layer.stride, padding=layer.padding, groups=layer.groups)
        
        if c_j.shape[2:] != activation.shape[2:]:
            c_j = F.interpolate(c_j, size=activation.shape[2:], mode='bilinear', align_corners=False)
            
        return activation * c_j

    def compute_relevance_and_activations(self, input_tensor):
        self.model.activations.clear() 
        inject_lrp_hooks(self.model)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_batch)
            relevance = output.clone()
            conv_layers = list(self.model.activations.keys())
            
            target_layer_idx = max(0, len(conv_layers) - 3)
            target_relevance = None
            target_act = None
            
            for i in range(len(conv_layers) - 1, -1, -1):
                layer = conv_layers[i]
                activation = self.model.activations[layer]
                relevance = self.lrp_alpha1_beta0(layer, activation, relevance)
                
                if i == target_layer_idx:
                    target_relevance = relevance.clone()
                    target_act = activation.clone()

        remove_lrp_hooks(self.model)
        self.model.activations.clear() 
        
        input_relevance_map = relevance[0].cpu().detach().numpy()
        if input_relevance_map.ndim == 3: 
            input_relevance_map = np.sum(np.abs(input_relevance_map), axis=0)
            
        border = 4
        input_relevance_map[:border, :] = 0; input_relevance_map[-border:, :] = 0
        input_relevance_map[:, :border] = 0; input_relevance_map[:, -border:] = 0

        vmax = np.percentile(input_relevance_map, 99.8) 
        vmin = np.min(input_relevance_map)

        if vmax > vmin:
            input_relevance_map = np.clip(input_relevance_map, vmin, vmax)
            input_relevance_map = (input_relevance_map - vmin) / (vmax - vmin)
        else:
            input_relevance_map = np.zeros_like(input_relevance_map)

        input_relevance_map = input_relevance_map ** 0.5
        return input_relevance_map, target_act, target_relevance

class NeuronSelector:
    @staticmethod
    def get_optimizer(forward_activation, backward_relevance, threshold=0.5):
        if forward_activation is None or backward_relevance is None: return None, None, None
        contributions = forward_activation * backward_relevance
        neuron_importance = torch.sum(contributions, dim=(2, 3)) 
        
        mean_val = torch.mean(neuron_importance)
        std_val = torch.std(neuron_importance)
        
        mask = neuron_importance > (mean_val + threshold * std_val)
        selected_neurons = mask.unsqueeze(-1).unsqueeze(-1) * forward_activation
        return selected_neurons, neuron_importance, mask

    @staticmethod
    def extract_k_paths(activation, mask, k=5):
        if activation is None or mask is None: return []
        selected_indices = torch.where(mask[0])[0]
        paths = []
        for idx in selected_indices[:k]:
            paths.append(activation[0, idx].cpu().detach().numpy())
        return paths

# ============================================================================
# 3. QUANTITATIVE METRICS & STATISTICS
# ============================================================================
def compute_reconstruction_metrics(full_output, selected_output):
    d_pre = full_output.cpu().detach().numpy().flatten()
    d_act = selected_output.cpu().detach().numpy().flatten()
    
    mse = np.mean((d_pre - d_act) ** 2)
    denominator = np.abs(d_pre) + np.abs(d_act)
    smape = 100 * np.mean(2 * np.abs(d_pre - d_act) / (denominator + 1e-10))
    return mse, smape

def compute_edge_concentration_score(heatmap):
    if heatmap is None or heatmap.size == 0: return 0.0
    heatmap = np.array(heatmap)
    if heatmap.max() > heatmap.min(): heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else: return 0.0
    
    flat = heatmap.flatten()
    flat_sum = flat.sum()
    if flat_sum < 1e-8: return 0.0
    
    flat_norm = flat / flat_sum
    with np.errstate(divide='ignore'): log_flat = np.log(flat_norm + 1e-10)
    entropy = -np.sum(flat_norm * log_flat)
    max_entropy = np.log(len(flat))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0

def compute_cohens_dz(data_proposed, data_baseline):
    """Computes Cohen's d_z for paired samples."""
    if len(data_proposed) < 2: return float('nan') # Prevent ddof=1 crash
    diff = np.array(data_proposed) - np.array(data_baseline)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0: return 0.0
    return np.mean(diff) / std_diff

def run_paired_ttest(prop_data, base_data):
    """Executes a paired t-test and returns the p-value and Cohen's d_z."""
    if len(prop_data) < 2 or len(base_data) < 2: return float('nan'), float('nan')
    t_stat, p_val = stats.ttest_rel(prop_data, base_data)
    d_z = compute_cohens_dz(prop_data, base_data)
    return p_val, d_z

def print_stat_row(metric_name, prop_data, base_data, base_name):
    """Helper to format and print statistical results."""
    p_val, d_z = run_paired_ttest(prop_data, base_data)
    if np.isnan(p_val):
        print(f"{metric_name:<20} | vs {base_name:<5} | Not enough data to compute statistics.")
        return
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    p_str = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
    print(f"{metric_name:<20} | vs {base_name:<5} | p: {p_str:<6} ({sig:<3}) | Cohen's d_z: {d_z:.4f}")

# ============================================================================
# 4. VISUALIZATION
# ============================================================================
def visualize_combined_k_paths(sample, relu_paths, frelu_paths, saga_paths, save_path):
    # Safety check: if a model failed to load, its paths list will be empty
    num_paths = min([len(p) for p in [relu_paths, frelu_paths, saga_paths] if len(p) > 0], default=0)
    if num_paths == 0: return
    
    fig, axes = plt.subplots(3, num_paths, figsize=(4 * num_paths, 12))
    if num_paths == 1: axes = np.expand_dims(axes, axis=1)

    def robust_normalize(feature_map, p_min=1.0, p_max=99.0):
        clean_map = feature_map.copy()
        border = 4
        clean_map[:border, :] = 0; clean_map[-border:, :] = 0
        clean_map[:, :border] = 0; clean_map[:, -border:] = 0
        vmax = np.percentile(clean_map, p_max); vmin = np.percentile(clean_map, p_min)
        if vmax > vmin: return (np.clip(clean_map, vmin, vmax) - vmin) / (vmax - vmin)
        return np.zeros_like(clean_map)

    for i in range(num_paths):
        if len(relu_paths) > i:
            relu_map = robust_normalize(relu_paths[i])
            im1 = axes[0, i].imshow(relu_map, cmap='hot')
            axes[0, i].set_title(f'Selected Path {i+1}', fontsize=12, fontweight='bold'); axes[0, i].axis('off')
            if i == 0: axes[0, i].text(-0.1, 0.5, 'ReLU Baseline', va='center', ha='right', rotation=90, transform=axes[0, i].transAxes, fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        if len(frelu_paths) > i:
            frelu_map = robust_normalize(frelu_paths[i])
            im2 = axes[1, i].imshow(frelu_map, cmap='hot')
            axes[1, i].axis('off')
            if i == 0: axes[1, i].text(-0.1, 0.5, 'FReLU Spatial', va='center', ha='right', rotation=90, transform=axes[1, i].transAxes, fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

        if len(saga_paths) > i:
            saga_map = robust_normalize(saga_paths[i])
            im3 = axes[2, i].imshow(saga_map, cmap='hot')
            axes[2, i].axis('off')
            if i == 0: axes[2, i].text(-0.1, 0.5, 'Proposed SAGA', va='center', ha='right', rotation=90, transform=axes[2, i].transAxes, fontsize=14, fontweight='bold')
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.suptitle(f'Deep-Layer K-Path Comparison (Bhati et al. 2025)\nSample: {str(sample["id"])[:40]}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_lrp_comparison(sample, relu_relevance, frelu_relevance, saga_relevance, relu_conc, frelu_conc, saga_conc, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    degraded = sample['degraded'].numpy().transpose(1, 2, 0)
    if degraded.shape[-1] == 1: degraded = degraded.squeeze(-1)
    degraded = np.clip((degraded + 1) / 2, 0, 1)
    
    axes[0].imshow(degraded, cmap='gray'); axes[0].set_title('Blurred Input', fontsize=12, fontweight='bold'); axes[0].axis('off')
    
    if relu_relevance is not None:
        im1 = axes[1].imshow(relu_relevance, cmap='hot', vmin=0, vmax=1); axes[1].set_title(f'ReLU LRP\nScore: {relu_conc:.4f}', fontsize=12, fontweight='bold'); axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if frelu_relevance is not None:
        im2 = axes[2].imshow(frelu_relevance, cmap='hot', vmin=0, vmax=1); axes[2].set_title(f'FReLU LRP\nScore: {frelu_conc:.4f}', fontsize=12, fontweight='bold'); axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    if saga_relevance is not None:
        im3 = axes[3].imshow(saga_relevance, cmap='hot', vmin=0, vmax=1); axes[3].set_title(f'SAGA LRP\nScore: {saga_conc:.4f}', fontsize=12, fontweight='bold'); axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'True LRP Analysis (Montavon et al. 2018)\nSample: {str(sample["id"])[:40]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 5. MAIN FUNCTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--dataset_root", type=str, default="./data/CT_dataset") # GitHub-safe default
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./lrp_results_Both-K-Path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k_paths", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("TRUE LRP ANALYSIS & NEURON SELECTION (For Reviewer #2)")
    print("Running Full Comparison: ReLU vs. FReLU vs. SAGA")
    print("=" * 80)

    # 1. Load Data
    if not os.path.exists(args.dataset_root):
        print(f"[ERROR] Dataset not found at: {args.dataset_root}. Please check your path.")
        return

    norm_mean = [0.5] * N_CHANNELS; norm_std = [0.5] * N_CHANNELS
    test_transform = transforms.Compose([transforms.Normalize(mean=norm_mean, std=norm_std)])
    test_dataset = MedicalImageRestorationDataset(root_dir=args.dataset_root, split='test', task_folder='data_dblur', degraded_folder='blur', transform=test_transform, target_size=TARGET_SIZE, input_channels=N_CHANNELS)
    
    if len(test_dataset) == 0:
        print("[ERROR] Test dataset is empty. Check paths.")
        return

    # 2. Load Models natively from the codebase (With existence checks)
    relu_model, frelu_model, saga_model = None, None, None
    relu_lrp, frelu_lrp, saga_lrp = None, None, None

    relu_path = os.path.join(args.model_dir, "DeblurResNet_CT_ReLU.pth") # Adjust filename to match your saving convention
    if os.path.exists(relu_path): 
        relu_model = DeblurResNet(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=nn.ReLU(), n_resblocks=DRN_N_RESBLOCKS, n_feats=DRN_N_FEATS).to(args.device)
        relu_model.load_state_dict(torch.load(relu_path, map_location=args.device, weights_only=True))
        relu_lrp = TrueLRP_Analyzer(relu_model, args.device)
    else: print(f"[WARNING] ReLU model weights not found at {relu_path}")
    
    frelu_path = os.path.join(args.model_dir, "DeblurResNet_CT_FReLU.pth")
    if os.path.exists(frelu_path): 
        frelu_model = DeblurResNet(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=FReLU(DRN_N_FEATS), n_resblocks=DRN_N_RESBLOCKS, n_feats=DRN_N_FEATS).to(args.device)
        frelu_model.load_state_dict(torch.load(frelu_path, map_location=args.device, weights_only=True))
        frelu_lrp = TrueLRP_Analyzer(frelu_model, args.device)
    else: print(f"[WARNING] FReLU model weights not found at {frelu_path}")

    saga_path = os.path.join(args.model_dir, "DeblurResNet_CT_SAGA.pth")
    if os.path.exists(saga_path): 
        saga_model = DeblurResNet(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=SAGA(DRN_N_FEATS), n_resblocks=DRN_N_RESBLOCKS, n_feats=DRN_N_FEATS).to(args.device)
        saga_model.load_state_dict(torch.load(saga_path, map_location=args.device, weights_only=True))
        saga_lrp = TrueLRP_Analyzer(saga_model, args.device)
    else: print(f"[WARNING] SAGA model weights not found at {saga_path}")
        
    indices = np.random.choice(len(test_dataset), min(args.num_samples, len(test_dataset)), replace=False)
    
    relu_scores, frelu_scores, saga_scores = [], [], []
    relu_all_mse, relu_all_smape = [], []
    frelu_all_mse, frelu_all_smape = [], []
    saga_all_mse, saga_all_smape = [], []
    
    print(f"\nProcessing {len(indices)} samples...")
    for i, idx in enumerate(indices):
        degraded, sharp, img_id = test_dataset[idx]
        sample = {'degraded': degraded, 'sharp': sharp, 'id': img_id}
        input_batch = degraded.unsqueeze(0).to(args.device)
        print(f"   Sample {i+1}: {str(img_id)[:30]}...")
        
        # Initialize loop variables to None
        relu_relevance = frelu_relevance = saga_relevance = None
        relu_conc = frelu_conc = saga_conc = 0.0
        relu_paths = frelu_paths = saga_paths = []

        try:
            if relu_lrp:
                relu_relevance, relu_target_act, relu_target_rel = relu_lrp.compute_relevance_and_activations(degraded)
                relu_conc = compute_edge_concentration_score(relu_relevance)
                relu_scores.append(relu_conc)
                relu_selected_act, _, relu_mask = NeuronSelector.get_optimizer(relu_target_act, relu_target_rel, threshold=0.5)
                
                with torch.no_grad():
                    relu_full_output = relu_model(input_batch)
                    relu_reconstructed_features = relu_model.tail(relu_selected_act)
                    if hasattr(relu_model, 'global_skip'): relu_reconstructed_features += relu_model.global_skip(input_batch)
                    else: relu_reconstructed_features += input_batch
                    relu_reconstructed_output = relu_model.final_activation(relu_reconstructed_features)
                
                relu_mse, relu_smape = compute_reconstruction_metrics(relu_full_output, relu_reconstructed_output)
                relu_all_mse.append(relu_mse); relu_all_smape.append(relu_smape)
                relu_paths = NeuronSelector.extract_k_paths(relu_target_act, relu_mask, k=args.k_paths)

            if frelu_lrp:
                frelu_relevance, frelu_target_act, frelu_target_rel = frelu_lrp.compute_relevance_and_activations(degraded)
                frelu_conc = compute_edge_concentration_score(frelu_relevance)
                frelu_scores.append(frelu_conc)
                frelu_selected_act, _, frelu_mask = NeuronSelector.get_optimizer(frelu_target_act, frelu_target_rel, threshold=0.5)
                
                with torch.no_grad():
                    frelu_full_output = frelu_model(input_batch)
                    frelu_reconstructed_features = frelu_model.tail(frelu_selected_act)
                    if hasattr(frelu_model, 'global_skip'): frelu_reconstructed_features += frelu_model.global_skip(input_batch)
                    else: frelu_reconstructed_features += input_batch
                    frelu_reconstructed_output = frelu_model.final_activation(frelu_reconstructed_features)
                
                frelu_mse, frelu_smape = compute_reconstruction_metrics(frelu_full_output, frelu_reconstructed_output)
                frelu_all_mse.append(frelu_mse); frelu_all_smape.append(frelu_smape)
                frelu_paths = NeuronSelector.extract_k_paths(frelu_target_act, frelu_mask, k=args.k_paths)

            if saga_lrp:
                saga_relevance, saga_target_act, saga_target_rel = saga_lrp.compute_relevance_and_activations(degraded)
                saga_conc = compute_edge_concentration_score(saga_relevance)
                saga_scores.append(saga_conc)
                saga_selected_act, _, saga_mask = NeuronSelector.get_optimizer(saga_target_act, saga_target_rel, threshold=0.5)
                
                with torch.no_grad():
                    saga_full_output = saga_model(input_batch)
                    saga_reconstructed_features = saga_model.tail(saga_selected_act)
                    if hasattr(saga_model, 'global_skip'): saga_reconstructed_features += saga_model.global_skip(input_batch)
                    else: saga_reconstructed_features += input_batch
                    saga_reconstructed_output = saga_model.final_activation(saga_reconstructed_features)
                
                saga_mse, saga_smape = compute_reconstruction_metrics(saga_full_output, saga_reconstructed_output)
                saga_all_mse.append(saga_mse); saga_all_smape.append(saga_smape)
                saga_paths = NeuronSelector.extract_k_paths(saga_target_act, saga_mask, k=args.k_paths)

            # Visualization
            if any([len(relu_paths), len(frelu_paths), len(saga_paths)]):
                visualize_combined_k_paths(sample, relu_paths, frelu_paths, saga_paths, 
                                           os.path.join(args.output_dir, f"sample_{i+1}_combined_k_paths.png"))
                
            visualize_lrp_comparison(sample, relu_relevance, frelu_relevance, saga_relevance, 
                                     relu_conc, frelu_conc, saga_conc, 
                                     os.path.join(args.output_dir, f"sample_{i+1}_lrp.png"))
            
        except Exception as e:
            print(f"      LRP computation failed for sample {i+1}: {e}")
            continue
            
        finally:
            # OOM FIX: Aggressive garbage collection for cpu users
            for var in ['relu_relevance', 'frelu_relevance', 'saga_relevance', 
                        'relu_target_act', 'frelu_target_act', 'saga_target_act', 
                        'relu_target_rel', 'frelu_target_rel', 'saga_target_rel',
                        'relu_selected_act', 'frelu_selected_act', 'saga_selected_act',
                        'input_batch', 'saga_full_output', 'frelu_full_output', 'relu_full_output']:
                if var in locals(): del locals()[var]
            gc.collect()
            torch.cuda.empty_cache()

    # ============================================================================
    # 6. STATISTICAL SIGNIFICANCE TESTING
    # ============================================================================
    print("\n" + "=" * 80)
    print("FINAL INTERPRETABILITY RESULTS & STATISTICAL SIGNIFICANCE (Reviewer #2)")
    print("=" * 80)
    
    # Only compute if SAGA actually loaded successfully
    if len(saga_scores) > 0:
        print("\n[A] Edge Concentration Scores (Higher is better)")
        if len(relu_scores) > 0: print(f"  ReLU:  {np.mean(relu_scores):.4f} ± {np.std(relu_scores):.4f}")
        if len(frelu_scores) > 0: print(f"  FReLU: {np.mean(frelu_scores):.4f} ± {np.std(frelu_scores):.4f}")
        print(f"  SAGA:  {np.mean(saga_scores):.4f} ± {np.std(saga_scores):.4f}")

        print("\n--- Paired t-tests: Edge Concentration (SAGA vs. Baselines) ---")
        if len(relu_scores) > 1: print_stat_row("Edge Conc", saga_scores, relu_scores, "ReLU")
        if len(frelu_scores) > 1: print_stat_row("Edge Conc", saga_scores, frelu_scores, "FReLU")

        if len(saga_all_mse) > 0:
            print("\n[B] Neuron Selection Validation (Lower is better)")
            if len(relu_all_mse) > 0: print(f"  ReLU  Recon MSE: {np.mean(relu_all_mse):.6f} | SMAPE: {np.mean(relu_all_smape):.2f}%")
            if len(frelu_all_mse) > 0: print(f"  FReLU Recon MSE: {np.mean(frelu_all_mse):.6f} | SMAPE: {np.mean(frelu_all_smape):.2f}%")
            print(f"  SAGA  Recon MSE: {np.mean(saga_all_mse):.6f} | SMAPE: {np.mean(saga_all_smape):.2f}%")
            
            print("\n--- Paired t-tests: Reconstruction Error (SAGA vs. Baselines) ---")
            if len(relu_all_mse) > 1: 
                print_stat_row("MSE", saga_all_mse, relu_all_mse, "ReLU")
                print_stat_row("SMAPE", saga_all_smape, relu_all_smape, "ReLU")
            if len(frelu_all_mse) > 1: 
                print_stat_row("MSE", saga_all_mse, frelu_all_mse, "FReLU")
                print_stat_row("SMAPE", saga_all_smape, frelu_all_smape, "FReLU")

        print("\n" + "=" * 80)
        print("Interpretation Guide:")
        print(" * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")
        print(" Cohen's d_z: 0.2 (Small), 0.5 (Medium), 0.8+ (Large Effect)")
        print(" Note: For MSE/SMAPE, a negative Cohen's d_z means SAGA is superior (lower error).")
        print("=" * 80)
    else:
        print("[ERROR] SAGA model data could not be processed. Cannot compute comparative statistics.")

if __name__ == "__main__":
    main()