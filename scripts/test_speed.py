import os, sys, argparse
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))
import utils
import cv2
import torch
from tqdm import tqdm
import numpy as np
from dataset import get_validation_data
from torch.utils.data import DataLoader
 
 
# ======================== Utilities ========================
 
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
 
def save_npy(f, save_dir, f_name):
    np.save(os.path.join(save_dir, f_name + '.npy'), f)
 
def save_img(f, save_dir, f_name):
    f = (f * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f_name + '.png'), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
 
def is_real_scene_path(*paths):
    return any('realscene' in os.path.normpath(str(path)).lower() for path in paths if path)

def tensor_to_img(t):
    return t.cpu().numpy().squeeze().transpose(1, 2, 0)

def get_model_size_mb(model):
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024
 
 
# ======================== Forward Pass With Complex Post-Processing Guard ========================
 
def forward_full(model, I_alpha, m_I, arch):
    """
    Run a full forward pass.
    Returns:
        dolp: [B, C, H, W] clamped to [0, 1]
        aolp: [B, C, H, W] range [0, pi]
        R:    [B, C, H, W] or None (DuRP only)
 
    Note: torch.abs / torch.angle operate on complex tensors and should
          run outside autocast, or be forced back to float32 precision.
          This function casts m_D back to complex64 before post-processing,
          so it is safe whether the caller uses autocast or not.
    """
    if arch == 'PRNet':
        k, m_A, m_D = model(I_alpha, m_I)
        R = None
    elif arch == 'DuRP':
        m_A, m_D, A_inf, R = model(I_alpha, m_I)
    else:
        raise ValueError(f"Unknown arch: {arch}")
 
    # Force float32 complex values to prevent autocast from lowering m_D to complex32.
    # torch.angle / torch.abs on complex32 can be unsupported or inaccurate on some GPUs.
    m_D_f32 = m_D.to(torch.complex64) if m_D.dtype != torch.complex64 else m_D
    dolp = torch.clamp(torch.abs(m_D_f32), 0, 1)
    aolp = torch.remainder(torch.angle(m_D_f32) * 0.5, 3.1415927)
    return dolp, aolp, R
 
 
# ======================== Model Loading ========================
 
def load_model(args, device, use_compile=False):
    """
    Always load model weights in FP32.
    FP16 inference is handled through autocast instead of model.half(),
    so precision-sensitive operations such as complex post-processing can
    still fall back safely.
    """
    model = utils.get_arch(args)
    utils.load_checkpoint(model, args.weights)
    model.cuda(device)
    model.eval()
    if use_compile:
        if not hasattr(torch, 'compile'):
            raise RuntimeError("torch.compile requires PyTorch >= 2.0")
        model = torch.compile(model, mode="reduce-overhead")
    return model
 
 
# ======================== Inference Core ========================
 
def run_inference(model, dataloader, device, arch, use_fp16,
                  save_results=False, result_dir=None):
    """
    Shared inference function for timing and optional result saving.
    FP16 uses autocast for mixed precision while keeping model weights in FP32.
    Returns:
        mean_ms (float), fps (float)
    """
    if save_results:
        result_dolp_dir = os.path.join(result_dir, 'dolp_R/')
        result_aolp_dir = os.path.join(result_dir, 'aolp_R/')
        ensure_dir(result_dolp_dir)
        ensure_dir(result_aolp_dir)
        if arch == 'DuRP':
            result_S0_dir = os.path.join(result_dir, 'S0_R/')
            ensure_dir(result_S0_dir)
 
    times = []
 
    with torch.no_grad():
        for data_val in dataloader:
            x1 = data_val[0].cuda(device)
            x2 = data_val[1].cuda(device)
            # Inputs do not need manual .half(); autocast handles casting.
 
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
 
            if use_fp16:
                # autocast accelerates the model forward pass with FP16.
                # forward_full casts back to complex64 for safe complex post-processing.
                with torch.cuda.amp.autocast():
                    dolp, aolp, R = forward_full(model, x1, x2, arch)
            else:
                dolp, aolp, R = forward_full(model, x1, x2, arch)
 
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
 
            # ---------- Optional Saving ----------
            if save_results:
                name      = data_val[2]
                current_h = data_val[3][0].item()
                current_w = data_val[3][1].item()
                prefix    = name[0]
 
                def resize_if_needed(t):
                    if t.shape[2] != current_h or t.shape[3] != current_w:
                        return torch.nn.functional.interpolate(
                            t, size=(current_h, current_w),
                            mode='bicubic', align_corners=False)
                    return t
 
                dolp = resize_if_needed(dolp)
                aolp = resize_if_needed(aolp)
                # forward_full guarantees float32-compatible outputs, so convert directly to numpy.
                save_npy(dolp.cpu().numpy().squeeze().transpose(1, 2, 0),
                         result_dolp_dir, prefix)
                save_npy(aolp.cpu().numpy().squeeze().transpose(1, 2, 0),
                         result_aolp_dir, prefix)
 
                if arch == 'DuRP' and R is not None:
                    R = resize_if_needed(R.float())
                    R_img = tensor_to_img(R)
                    save_img(R_img, result_S0_dir, prefix)
 
    times   = np.array(times)
    mean_ms = float(np.mean(times))
    fps     = 1000.0 / mean_ms
    return mean_ms, fps
 
 
def warmup(model, dummy1, dummy2, device, arch, use_fp16, n=20):
    """Warm up the GPU with the same autocast path used for inference."""
    with torch.no_grad():
        for _ in range(n):
            x1 = dummy1.cuda(device)
            x2 = dummy2.cuda(device)
            if use_fp16:
                with torch.cuda.amp.autocast():
                    _ = forward_full(model, x1, x2, arch)
            else:
                _ = forward_full(model, x1, x2, arch)
    torch.cuda.synchronize()


# ======================== Main ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Polarization Reconstruction Inference Benchmark')
    parser.add_argument('--val_dir',    default=None, type=str)
    parser.add_argument('--result_dir', default=None, type=str)
    parser.add_argument('--weights',    default=None, type=str)
    parser.add_argument('--gpu_device', default='0', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--arch',       default='DuRP', type=str, choices=['PRNet', 'DuRP'])
    parser.add_argument('--stage',      default='test', type=str)

    # Core options
    parser.add_argument('--mode', default='fp16', type=str,
                        choices=['fp32', 'fp16', 'fp16_compile'],
                        help='Inference mode: fp32 | fp16 | fp16_compile')
    parser.add_argument('--save', action='store_true',
                        help='Save inference results (dolp / aolp / S0)')

    args = parser.parse_args()

    utils.mkdir(args.result_dir)
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")

    # ---------- Mode Parsing ----------
    use_fp16    = args.mode in ('fp16', 'fp16_compile')
    use_compile = args.mode == 'fp16_compile'
    n_warmup    = 30 if use_compile else 20

    # ---------- Load Model (Always FP32 Weights) ----------
    model = load_model(args, device, use_compile=use_compile)
    model_size_mb = get_model_size_mb(model)

    # ---------- Data Loading ----------
    test_dataset = get_validation_data(args)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=8, drop_last=False)

    first_batch = next(iter(test_loader))
    dummy_in1   = first_batch[0].clone()
    dummy_in2   = first_batch[1].clone()

    # ---------- Warmup ----------
    warmup(model, dummy_in1, dummy_in2, device, args.arch, use_fp16, n=n_warmup)

    # ---------- Inference + Timing ----------
    mean_ms, fps = run_inference(
        model, test_loader, device, args.arch,
        use_fp16=use_fp16,
        save_results=args.save,
        result_dir=args.result_dir if args.save else None
    )

    # ---------- Console Output ----------
    print(f"\n{'='*45}")
    print(f"  Mode        : {args.mode.upper()}")
    print(f"  Model Size  : {model_size_mb:.2f} MB")
    print(f"  Latency     : {mean_ms:.2f} ms / image")
    print(f"  FPS         : {fps:.2f}")
    if args.save:
        print(f"  Results saved to: {args.result_dir}")
    print(f"{'='*45}\n")
