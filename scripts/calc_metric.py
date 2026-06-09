import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

def peridic_psnr(aolp_gt, aolp_result):
    """
    Calculate the PSNR of aolp.
    :param aolp_gt: GT [0-pi]
    :param aolp_result: result [0-pi]
    :return: PSNR value
    """
    aolp_diff_1 = np.abs(aolp_gt - aolp_result)
    aolp_diff_2 = np.pi - np.abs(aolp_gt - aolp_result)
    aolp_diff = np.minimum(aolp_diff_1, aolp_diff_2)
    aolp_rmse = np.mean(aolp_diff ** 2)
    
    if aolp_rmse == 0:
        return float('inf')
        
    aolp_psnr = 10 * np.log10(np.pi ** 2 / aolp_rmse)
    return aolp_psnr

def main():
    parser = argparse.ArgumentParser(description='Calculate PSNR and SSIM for Polarization Images')
    parser.add_argument('--gt_dir', default='./datas/SRPG/test/GT/', type=str, help='Directory of Ground Truth')
    parser.add_argument('--pred_dir', default='./results/DuRP/SRPG/FP16_compile/', type=str, help='Directory of Predicted Results')
    args = parser.parse_args()
    sub_dirs = {'S0': 'S0_R', 'DoLP': 'dolp_R', 'AoLP': 'aolp_R'}
    

    s0_gt_path = os.path.join(args.gt_dir, sub_dirs['S0'])
    if not os.path.exists(s0_gt_path):
        raise FileNotFoundError(f"GT Directory not found: {s0_gt_path}")
        
    img_names = [f for f in os.listdir(s0_gt_path) if f.endswith('.png')]
    
    metrics = {
        'S0_PSNR': [], 'S0_SSIM': [],
        'DoLP_PSNR': [], 'DoLP_SSIM': [],
        'AoLP_PSNR': [], 'AoLP_SSIM': []
    }

    for img_name in tqdm(img_names, desc="Calculating Metrics"):
            base_name = img_name.replace('.png', '')
            
            s0_gt_file = os.path.join(args.gt_dir, sub_dirs['S0'], img_name)
            s0_pred_file = os.path.join(args.pred_dir, sub_dirs['S0'], img_name)
            
            if not os.path.exists(s0_pred_file):
                print(f"\n[Warning] Prediction file missing, skipping: {s0_pred_file}")
                continue

            s0_gt = cv2.imread(s0_gt_file, cv2.IMREAD_COLOR)
            s0_pred = cv2.imread(s0_pred_file, cv2.IMREAD_COLOR)
            
            if s0_pred is None:
                print(f"\n[Error] Failed to read prediction file (corrupted?), skipping: {s0_pred_file}")
                continue

            s0_gt = cv2.cvtColor(s0_gt, cv2.COLOR_BGR2RGB)
            s0_pred = cv2.cvtColor(s0_pred, cv2.COLOR_BGR2RGB)

            metrics['S0_PSNR'].append(calc_psnr(s0_gt, s0_pred, data_range=255))
            metrics['S0_SSIM'].append(calc_ssim(s0_gt, s0_pred, data_range=255, channel_axis=-1))

            dolp_name = base_name + '.npy'
            dolp_gt_file = os.path.join(args.gt_dir, sub_dirs['DoLP'], dolp_name)
            dolp_pred_file = os.path.join(args.pred_dir, sub_dirs['DoLP'], dolp_name)
            
            if not os.path.exists(dolp_pred_file):
                print(f"\n[Warning] DoLP prediction missing, skipping: {dolp_pred_file}")
                continue

            dolp_gt = np.clip(np.load(dolp_gt_file), 0, 1)
            dolp_pred = np.clip(np.load(dolp_pred_file), 0, 1)

            dolp_kwargs = {'data_range': 1.0}
            if dolp_gt.ndim == 3:
                dolp_kwargs['channel_axis'] = -1

            metrics['DoLP_PSNR'].append(calc_psnr(dolp_gt, dolp_pred, data_range=1.0))
            metrics['DoLP_SSIM'].append(calc_ssim(dolp_gt, dolp_pred, **dolp_kwargs))

            aolp_name = base_name + '.npy'
            aolp_gt_file = os.path.join(args.gt_dir, sub_dirs['AoLP'], aolp_name)
            aolp_pred_file = os.path.join(args.pred_dir, sub_dirs['AoLP'], aolp_name)
            
            if not os.path.exists(aolp_pred_file):
                print(f"\n[Warning] AoLP prediction missing, skipping: {aolp_pred_file}")
                continue

            aolp_gt = np.load(aolp_gt_file)
            aolp_pred = np.load(aolp_pred_file)
            # aolp_pred = np.load(aolp_pred_file) * np.pi # swin and direct

            aolp_kwargs = {'data_range': np.pi}
            if aolp_gt.ndim == 3:
                aolp_kwargs['channel_axis'] = -1

            metrics['AoLP_PSNR'].append(peridic_psnr(aolp_gt, aolp_pred))
            metrics['AoLP_SSIM'].append(calc_ssim(aolp_gt, aolp_pred, **aolp_kwargs))

    print("\n" + "="*40)
    print(" "*10 + "Evaluation Results")
    print("="*40)
    print(f"S0   - PSNR: {np.mean(metrics['S0_PSNR']):.4f} dB, SSIM: {np.mean(metrics['S0_SSIM']):.4f}")
    print(f"DoLP - PSNR: {np.mean(metrics['DoLP_PSNR']):.4f} dB, SSIM: {np.mean(metrics['DoLP_SSIM']):.4f}")
    print(f"AoLP - PSNR: {np.mean(metrics['AoLP_PSNR']):.4f} dB, SSIM: {np.mean(metrics['AoLP_SSIM']):.4f}")
    print("="*40)

if __name__ == '__main__':
    main()
