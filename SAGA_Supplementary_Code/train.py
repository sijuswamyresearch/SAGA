# -*- coding: utf-8 -*-
import os
import glob
import time
import copy
import random
import argparse
import pickle
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from collections import defaultdict
import math

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import Models and Activations
from models.unet import UNet
from models.resnet import DeblurResNet
from models.edsr import EDSR_Deblur
from models.vggnet import PlainVGGNet  # <--- NEW IMPORT
from models.saga_layer import *
from evaluate import calculate_all_metrics, save_comparison_plot, analyze_frequency_and_spatial, plot_loss_curves, evaluate_model_on_test_set, run_anova_analysis, ActivationAnalyzer

# --- Configuration ---
MODEL_CHOICE = "PlainVGGNet" # Options: "UNET", "EDSR", "DeblurResNet", "PlainVGGNet"
DATASET_NAME = "CT_dataset"
TASK_NAME = "data_dblur"
DEGRADED_FOLDER_NAME = "blur"
DATASET_PARENT_DIR = "/dist_home/siju/AFS/"

N_CHANNELS = 3 if DATASET_NAME == "HAM10000" else 1

EXPERIMENT_FOLDER_NAME = f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_activation_analysis"
DATASET_ROOT = os.path.join(DATASET_PARENT_DIR, DATASET_NAME)
MODEL_DIR = os.path.join("models", EXPERIMENT_FOLDER_NAME)
RESULT_DIR = os.path.join("results", EXPERIMENT_FOLDER_NAME)

BATCH_SIZE = 32
EPOCHS = 30
TARGET_SIZE = (256, 256)
NUM_RUNS = 1
GLOBAL_WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15

HPO_ENABLED = OPTUNA_AVAILABLE and False 
HPO_N_TRIALS = 10
HPO_N_EPOCHS = 3
DEFAULT_LR = 1e-4
LR_RANGE = [1e-5, 5e-4]

EDSR_N_RESBLOCKS = 16
EDSR_N_FEATS = 64
EDSR_RES_SCALE = 0.1
DRN_N_RESBLOCKS = 16
DRN_N_FEATS = 64

ACTIVATIONS_FOR_FOCUSED_STUDY = {
    "ReLU": {"fn": nn.ReLU()},
    "Swish": {"fn": Swish()},
    "FReLU": {"fn": FReLU(channels=64)}, # Ensure channel arg matches your network feats
    "SAGA": {"fn": SAGA(channels=64)}
}
ALL_ACTIVATIONS_CONFIG = ACTIVATIONS_FOR_FOCUSED_STUDY

# --- Medical Image Dataset ---
class MedicalImageRestorationDataset(Dataset):
    def __init__(self, root_dir, split='train', task_folder='data_dblur', degraded_folder='blur', transform=None, target_size=None, input_channels=3):
        self.task_split_path = os.path.join(root_dir, task_folder, split)
        self.sharp_dir = os.path.join(self.task_split_path, 'sharp')
        self.degraded_dir = os.path.join(self.task_split_path, degraded_folder)
        self.transform = transform
        self.target_size = target_size
        self.input_channels = input_channels

        self.sharp_files = sorted(glob.glob(os.path.join(self.sharp_dir, '*.pt')))
        if not self.sharp_files: raise FileNotFoundError(f"No sharp tensors found in {self.sharp_dir}")

        self.image_pairs = []
        self.image_ids = []
        for sharp_path in self.sharp_files:
            filename = os.path.basename(sharp_path)
            degraded_path = os.path.join(self.degraded_dir, filename)
            if os.path.isfile(degraded_path):
                self.image_pairs.append((sharp_path, degraded_path))
                self.image_ids.append(filename)
        if not self.image_pairs: raise FileNotFoundError("No valid tensor pairs found")

    def __len__(self): return len(self.image_pairs)

    def __getitem__(self, idx):
        sharp_path, degraded_path = self.image_pairs[idx]
        image_id = self.image_ids[idx]
        current_h, current_w = self.target_size if self.target_size else (256, 256)

        try:
            try: sharp_tensor = torch.load(sharp_path, weights_only=False).float(); degraded_tensor = torch.load(degraded_path, weights_only=False).float()
            except TypeError: sharp_tensor = torch.load(sharp_path).float(); degraded_tensor = torch.load(degraded_path).float()

            if sharp_tensor.dim() == 3 and sharp_tensor.shape[2] == self.input_channels: sharp_tensor = sharp_tensor.permute(2, 0, 1)
            elif sharp_tensor.dim() == 2 and self.input_channels == 1: sharp_tensor = sharp_tensor.unsqueeze(0)
            if degraded_tensor.dim() == 3 and degraded_tensor.shape[2] == self.input_channels: degraded_tensor = degraded_tensor.permute(2, 0, 1)
            elif degraded_tensor.dim() == 2 and self.input_channels == 1: degraded_tensor = degraded_tensor.unsqueeze(0)

            sharp_tensor = torch.clamp(sharp_tensor, 0.0, 1.0); degraded_tensor = torch.clamp(degraded_tensor, 0.0, 1.0)
            if not self.target_size: current_h, current_w = sharp_tensor.shape[1], sharp_tensor.shape[2]
        except Exception:
            return torch.zeros((self.input_channels, current_h, current_w)), torch.zeros((self.input_channels, current_h, current_w)), f"error_load_{image_id}"

        if self.target_size:
            resize_transform = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            try: sharp_tensor = resize_transform(sharp_tensor); degraded_tensor = resize_transform(degraded_tensor)
            except Exception: return torch.zeros((self.input_channels, current_h, current_w)), torch.zeros((self.input_channels, current_h, current_w)), f"error_resize_{image_id}"

        if self.transform:
            apply_hflip = False
            if any(isinstance(t, transforms.RandomHorizontalFlip) for t in getattr(self.transform, 'transforms', [self.transform])):
                if random.random() > 0.5: apply_hflip = True

            temp_transform_list = [t for t in getattr(self.transform, 'transforms', [self.transform]) if not isinstance(t, transforms.RandomHorizontalFlip)]
            if apply_hflip:
                sharp_tensor = transforms.functional.hflip(sharp_tensor)
                degraded_tensor = transforms.functional.hflip(degraded_tensor)

            if temp_transform_list:
                composed = transforms.Compose(temp_transform_list)
                sharp_tensor = composed(sharp_tensor); degraded_tensor = composed(degraded_tensor)
        return degraded_tensor, sharp_tensor, image_id


def objective(trial, act_name_hpo, act_config_hpo, hpo_train_loader, hpo_val_loader, device_hpo):
    lr_hpo = trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)
    activation_fn_hpo_template = copy.deepcopy(act_config_hpo["fn"])

    # <--- NEW ROUTING LOGIC --->
    if MODEL_CHOICE == "EDSR": 
        model_hpo = EDSR_Deblur(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=activation_fn_hpo_template).to(device_hpo)
    elif MODEL_CHOICE == "DeblurResNet": 
        model_hpo = DeblurResNet(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=activation_fn_hpo_template, n_resblocks=DRN_N_RESBLOCKS, n_feats=DRN_N_FEATS).to(device_hpo)
    elif MODEL_CHOICE == "PlainVGGNet": 
        model_hpo = PlainVGGNet(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=activation_fn_hpo_template, n_layers=18, n_feats=64).to(device_hpo)
    else: 
        model_hpo = UNet(n_channels=N_CHANNELS, n_classes=N_CHANNELS, activation_fn=activation_fn_hpo_template).to(device_hpo)

    optimizer_hpo = optim.Adam(model_hpo.parameters(), lr=lr_hpo, weight_decay=GLOBAL_WEIGHT_DECAY)
    criterion_hpo = nn.L1Loss(); scaler_hpo = GradScaler(enabled=(device_hpo.type == 'cuda'))
    
    model_hpo.train()
    for epoch_hpo in range(HPO_N_EPOCHS):
        num_batch_train_hpo = 0
        for batch_data_hpo in hpo_train_loader:
            degraded_hpo, sharp_hpo, _ = batch_data_hpo 
            degraded_hpo, sharp_hpo = degraded_hpo.to(device_hpo), sharp_hpo.to(device_hpo)
            optimizer_hpo.zero_grad(set_to_none=True)
            with autocast(enabled=(device_hpo.type == 'cuda')): 
                output_hpo = model_hpo(degraded_hpo)
                loss_hpo = criterion_hpo(output_hpo, sharp_hpo)
            if torch.isnan(loss_hpo) or torch.isinf(loss_hpo): raise optuna.TrialPruned()
            scaler_hpo.scale(loss_hpo).backward(); scaler_hpo.step(optimizer_hpo); scaler_hpo.update()
            if num_batch_train_hpo >= 10: break 
    model_hpo.eval(); val_loss_hpo_accum = 0; num_batches_hpo_val = 0
    with torch.no_grad():
        for batch_data_val_hpo in hpo_val_loader:
            degraded_val_hpo, sharp_val_hpo, _ = batch_data_val_hpo 
            degraded_val_hpo, sharp_val_hpo = degraded_val_hpo.to(device_hpo), sharp_val_hpo.to(device_hpo)
            with autocast(enabled=(device_hpo.type == 'cuda')): 
                output_val_hpo = model_hpo(degraded_val_hpo)
                v_loss_hpo = criterion_hpo(output_val_hpo, sharp_val_hpo).item()
            if not np.isnan(v_loss_hpo) and not np.isinf(v_loss_hpo): val_loss_hpo_accum += v_loss_hpo; num_batches_hpo_val += 1
            if num_batches_hpo_val >= 5: break 
    avg_val_loss_hpo = val_loss_hpo_accum / num_batches_hpo_val if num_batches_hpo_val > 0 else float('inf')
    if np.isnan(avg_val_loss_hpo) or avg_val_loss_hpo == float('inf'): raise optuna.TrialPruned()
    trial.report(avg_val_loss_hpo, HPO_N_EPOCHS - 1) 
    if trial.should_prune(): raise optuna.TrialPruned()
    return avg_val_loss_hpo

def train_and_evaluate(act_name_train, act_config_train, learning_rate_train, weight_decay_train, train_loader_main, val_loader_main, run_idx_train, device_train, result_dir_train=RESULT_DIR, model_dir_train=MODEL_DIR):
    print(f"\n=== Training {MODEL_CHOICE} Run {run_idx_train+1} with {act_name_train} (LR={learning_rate_train:.2e}) ===")
    current_activation_fn_template = copy.deepcopy(act_config_train["fn"])
    
    # <--- NEW ROUTING LOGIC --->
    if MODEL_CHOICE == "EDSR": 
        model_train = EDSR_Deblur(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=current_activation_fn_template).to(device_train)
    elif MODEL_CHOICE == "DeblurResNet": 
        model_train = DeblurResNet(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=current_activation_fn_template, n_resblocks=DRN_N_RESBLOCKS, n_feats=DRN_N_FEATS).to(device_train)
    elif MODEL_CHOICE == "PlainVGGNet": 
        model_train = PlainVGGNet(n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS, activation_fn_template=current_activation_fn_template, n_layers=18, n_feats=64).to(device_train)
    else: 
        model_train = UNet(n_channels=N_CHANNELS, n_classes=N_CHANNELS, activation_fn=current_activation_fn_template).to(device_train)

    analyzer_train = ActivationAnalyzer(model_train)
    optimizer_train = optim.Adam(model_train.parameters(), lr=learning_rate_train, weight_decay=weight_decay_train)
    scheduler_train = ReduceLROnPlateau(optimizer_train, 'min', patience=7, factor=0.5, verbose=False) 
    criterion_train = nn.L1Loss(); use_amp_train = (device_train.type == 'cuda'); scaler_train = GradScaler(enabled=use_amp_train)
    
    train_loss_history_ep = []; val_loss_history_ep = []
    best_val_loss_ep = float('inf'); epochs_no_improve_ep = 0; best_model_state_ep = None
    vis_blurred_val_tensor, vis_sharp_val_tensor = None, None 

    try:
        vis_data_loader = DataLoader(val_loader_main.dataset, batch_size=1, shuffle=False)
        vis_batch_content = next(iter(vis_data_loader))
        vis_blurred_val_tensor, vis_sharp_val_tensor = vis_batch_content[0][0], vis_batch_content[1][0]
    except Exception as e_vis: print(f"Warning: Error getting visualization batch: {e_vis}")

    for epoch_train in range(EPOCHS):
        model_train.train(); epoch_loss_accum = 0; num_train_batches_ep = 0
        pbar_train_ep = tqdm(train_loader_main, desc=f"Epoch {epoch_train+1}/{EPOCHS} [Train]", leave=False)
        for batch_data_train_ep in pbar_train_ep:
            blurred_ep, sharp_ep, _ = batch_data_train_ep 
            blurred_ep, sharp_ep = blurred_ep.to(device_train), sharp_ep.to(device_train)
            optimizer_train.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp_train): 
                output_ep = model_train(blurred_ep)
                loss_ep = criterion_train(output_ep, sharp_ep)
            if torch.isnan(loss_ep) or torch.isinf(loss_ep): continue
            scaler_train.scale(loss_ep).backward(); scaler_train.unscale_(optimizer_train); torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=1.0)
            scaler_train.step(optimizer_train); scaler_train.update()
            epoch_loss_accum += loss_ep.item(); num_train_batches_ep += 1; pbar_train_ep.set_postfix(loss=f"{loss_ep.item():.4f}")
        
        avg_train_loss_ep = epoch_loss_accum / num_train_batches_ep if num_train_batches_ep > 0 else float('nan'); train_loss_history_ep.append(avg_train_loss_ep)
        model_train.eval(); val_loss_accum_ep = 0; num_val_batches_ep = 0; epoch_val_metrics_lists = defaultdict(list)
        pbar_val_ep = tqdm(val_loader_main, desc=f"Epoch {epoch_train+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for batch_data_val_ep in pbar_val_ep:
                blurred_val_ep, sharp_val_ep, _ = batch_data_val_ep
                blurred_val_ep, sharp_val_ep = blurred_val_ep.to(device_train), sharp_val_ep.to(device_train)
                with autocast(enabled=use_amp_train): 
                    output_val_ep = model_train(blurred_val_ep)
                    v_loss_ep = criterion_train(output_val_ep, sharp_val_ep)
                if not torch.isnan(v_loss_ep) and not torch.isinf(v_loss_ep):
                    val_loss_accum_ep += v_loss_ep.item(); num_val_batches_ep += 1
                    if blurred_val_ep.size(0) > 0:
                         metrics_batch_first_img = calculate_all_metrics(sharp_val_ep[0].float(), output_val_ep[0].float(), device_train, N_CHANNELS)
                         for key_metric, val_metric in metrics_batch_first_img.items(): epoch_val_metrics_lists[key_metric].append(val_metric)
                pbar_val_ep.set_postfix(loss=f"{v_loss_ep.item() if not (torch.isnan(v_loss_ep) or torch.isinf(v_loss_ep)) else float('nan'):.4f}")
        
        avg_val_loss_ep = val_loss_accum_ep / num_val_batches_ep if num_val_batches_ep > 0 else float('inf'); val_loss_history_ep.append(avg_val_loss_ep)
        avg_epoch_val_metrics_print = {f"{k}_mean": np.mean([m for m in v if np.isfinite(m)]) for k, v in epoch_val_metrics_lists.items()}
        print(f"Epoch {epoch_train+1}/{EPOCHS} - Train: {avg_train_loss_ep:.5f}, Val: {avg_val_loss_ep:.5f} (Val PSNR: {avg_epoch_val_metrics_print.get('psnr_mean', 0.0):.2f})")
        
        scheduler_train.step(avg_val_loss_ep)
        if avg_val_loss_ep < best_val_loss_ep:
            best_val_loss_ep = avg_val_loss_ep; best_model_state_ep = copy.deepcopy(model_train.state_dict()); epochs_no_improve_ep = 0
            save_path_model = os.path.join(model_dir_train, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name_train}_run{run_idx_train+1}_best.pth")
            torch.save(best_model_state_ep, save_path_model)
        else: epochs_no_improve_ep += 1
        if epochs_no_improve_ep >= EARLY_STOPPING_PATIENCE: break
    
    plot_loss_curves(train_loss_history_ep, val_loss_history_ep, act_name_train, run_idx_train, MODEL_CHOICE, DATASET_NAME, TASK_NAME, result_dir=result_dir_train)
    model_train.load_state_dict(best_model_state_ep); model_train.eval()
    
    final_val_metrics_all_imgs = defaultdict(list); final_val_loss_accum = 0; final_val_batches_count = 0
    with torch.no_grad():
        for batch_data_final_val in val_loader_main:
            blurred_final, sharp_final, _ = batch_data_final_val
            blurred_final, sharp_final = blurred_final.to(device_train), sharp_final.to(device_train)
            with autocast(enabled=use_amp_train): 
                output_final = model_train(blurred_final)
                v_loss_final = criterion_train(output_final, sharp_final)
            if not torch.isnan(v_loss_final) and not torch.isinf(v_loss_final):
                final_val_loss_accum += v_loss_final.item(); final_val_batches_count += 1
                for j_img in range(blurred_final.size(0)):
                     metrics_per_img = calculate_all_metrics(sharp_final[j_img].float(), output_final[j_img].float(), device_train, N_CHANNELS)
                     for k, v in metrics_per_img.items(): final_val_metrics_all_imgs[k].append(v)
    
    avg_final_val_metrics_summary = {}
    for k, v in final_val_metrics_all_imgs.items():
         valid_sum = [val for val in v if np.isfinite(val)]
         avg_final_val_metrics_summary[f"{k}_mean"] = np.mean(valid_sum) if valid_sum else float('nan')
         avg_final_val_metrics_summary[f"{k}_std"] = np.std(valid_sum) if len(valid_sum) > 1 else 0.0
    
    avg_final_val_loss_summary = final_val_loss_accum / final_val_batches_count if final_val_batches_count > 0 else float('inf')
    summary_results_dict = {
        "psnr_mean": avg_final_val_metrics_summary.get("psnr_mean", float('nan')), "psnr_std": avg_final_val_metrics_summary.get("psnr_std", float('nan')),
        "ssim_mean": avg_final_val_metrics_summary.get("ssim_mean", float('nan')), "ssim_std": avg_final_val_metrics_summary.get("ssim_std", float('nan')),
        "epi_mean": avg_final_val_metrics_summary.get("epi_mean", float('nan')), "epi_std": avg_final_val_metrics_summary.get("epi_std", float('nan')),
        "hf_recon_mean": avg_final_val_metrics_summary.get("hf_recon_mean", float('nan')), "hf_recon_std": avg_final_val_metrics_summary.get("hf_recon_std", float('nan')),
        "lpips_mean": avg_final_val_metrics_summary.get("lpips_mean", float('nan')), "lpips_std": avg_final_val_metrics_summary.get("lpips_std", float('nan')),
        "final_train_loss": train_loss_history_ep[-1] if train_loss_history_ep and np.isfinite(train_loss_history_ep[-1]) else float('nan'),
        "final_val_loss": avg_final_val_loss_summary if np.isfinite(avg_final_val_loss_summary) else float('nan'),
        "inf_time_ms": float('nan') 
    }
    
    if vis_blurred_val_tensor is not None and vis_sharp_val_tensor is not None:
        with torch.no_grad():
            vis_input_tensor = vis_blurred_val_tensor.unsqueeze(0).to(device_train)
            with autocast(enabled=use_amp_train): vis_output_val_tensor_dev = model_train(vis_input_tensor)
        save_comparison_plot(vis_blurred_val_tensor.cpu(), vis_output_val_tensor_dev[0].cpu(), vis_sharp_val_tensor.cpu(), act_name_train, run_idx_train, summary_results_dict, MODEL_CHOICE, DATASET_NAME, TASK_NAME, N_CHANNELS, result_dir=result_dir_train)
        analyze_frequency_and_spatial(model_train, analyzer_train, vis_sharp_val_tensor.cpu(), vis_output_val_tensor_dev[0].cpu(), vis_blurred_val_tensor.cpu(), act_name_train, run_idx_train, MODEL_CHOICE, DATASET_NAME, TASK_NAME, N_CHANNELS, result_dir=result_dir_train)
    
    return summary_results_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation Function Analysis for Medical Images")
    parser.add_argument("--job_part", type=int, default=None, help="Specify which part of the activation functions to run.")
    parser.add_argument("--total_jobs", type=int, default=2, help="Total number of jobs the workload is split into.")
    parser.add_argument("--combine_results", action="store_true", help="If set, skips training and combines results.")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    for folder in ["metrics", "spatial_analysis", "fft_analysis", "comparison_plots", "loss_curves"]:
        os.makedirs(os.path.join(RESULT_DIR, folder), exist_ok=True)

    if args.combine_results:
        import evaluate
        evaluate.combine_and_run_anova(args.total_jobs, RESULT_DIR, EXPERIMENT_FOLDER_NAME, ALL_ACTIVATIONS_CONFIG.keys(), MODEL_CHOICE, DATASET_NAME, TASK_NAME, NUM_RUNS)
        exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    norm_mean = [0.5] * N_CHANNELS; norm_std = [0.5] * N_CHANNELS 
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Normalize(mean=norm_mean, std=norm_std)])
    test_transform = transforms.Compose([transforms.Normalize(mean=norm_mean, std=norm_std)])
    
    train_dataset = MedicalImageRestorationDataset(DATASET_ROOT, 'train', TASK_NAME, DEGRADED_FOLDER_NAME, train_transform, TARGET_SIZE, N_CHANNELS)
    val_dataset = MedicalImageRestorationDataset(DATASET_ROOT, 'val', TASK_NAME, DEGRADED_FOLDER_NAME, test_transform, TARGET_SIZE, N_CHANNELS)
    test_dataset = MedicalImageRestorationDataset(DATASET_ROOT, 'test', TASK_NAME, DEGRADED_FOLDER_NAME, test_transform, TARGET_SIZE, N_CHANNELS)
    
    num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() > 1 else 1)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=num_workers)

    activations_to_run = ALL_ACTIVATIONS_CONFIG
    if args.job_part is not None:
        keys = list(ALL_ACTIVATIONS_CONFIG.keys())
        chunk = math.ceil(len(keys) / args.total_jobs)
        activations_to_run = {k: ALL_ACTIVATIONS_CONFIG[k] for k in keys[(args.job_part-1)*chunk:args.job_part*chunk]}

    all_run_aggregated_val = {run: {} for run in range(NUM_RUNS)}
    all_test_image_results = []
    
    for run_idx in range(NUM_RUNS):
        for act_name, act_config in activations_to_run.items():
            best_lr = DEFAULT_LR
            if HPO_ENABLED: 
                pass # Optuna Logic execution

            val_summary = train_and_evaluate(act_name, act_config, best_lr, GLOBAL_WEIGHT_DECAY, train_loader, val_loader, run_idx, device)
            all_run_aggregated_val[run_idx][act_name] = val_summary
            
            best_model_path = os.path.join(MODEL_DIR, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name}_run{run_idx+1}_best.pth")
            if os.path.exists(best_model_path):
                test_results = evaluate_model_on_test_set(best_model_path, act_config["fn"], test_loader, device, act_name, run_idx, MODEL_CHOICE, DATASET_NAME, TASK_NAME, N_CHANNELS, DRN_N_RESBLOCKS, DRN_N_FEATS)
                all_test_image_results.extend(test_results)
    
    job_part_suffix = f"_part{args.job_part}" if args.job_part is not None else ""
    import pandas as pd
    if all_test_image_results:
        df = pd.DataFrame(all_test_image_results)
        df.to_csv(os.path.join(RESULT_DIR, "metrics", f"{EXPERIMENT_FOLDER_NAME}_per_image_test_metrics{job_part_suffix}.csv"), index=False)