<div align="center">

## <font color="#2e6da4" style="font-size: 16px;">DuRP: Dual-Stage Physics-Embedded Learning for<br>Joint Radiance and Polarization Restoration</font>

Zhenshuo Yang <sup>1,2</sup>,
Qian He <sup>1,2</sup>,
Zhiyuan Liu <sup>1,2</sup>,
[Baojie Fan](mailto:jobfbj@gmail.com) <sup>3*</sup>,
[Jiandong Tian](mailto:tianjd@sia.cn) <sup>1*</sup>

<p align="center">
  <sup>1</sup> State Key Laboratory of Robotics and Intelligent Systems,
  Shenyang Institute of Automation, Chinese Academy of Sciences <br>
  <sup>2</sup> University of the Chinese Academy of Sciences <br>
  <sup>3</sup> Nanjing University of Posts and Telecommunications
</p>

<p align="center">
  <small><sup>*</sup> Corresponding authors.</small>
</p>

[![Paper](https://img.shields.io/badge/ICML_2026-Paper-red?logo=arxiv&logoColor=red)](https://your-paper-link.pdf)
[![License](https://img.shields.io/badge/License-Research_Use_Only-green)](#)

</div>

**Abstract:** Polarization information is valuable for many computer vision applications.
However, in hazy environments, polarization information is severely attenuated due to
the degradation of captured polarized images. Existing dehazing methods struggle to
effectively restore polarization information, as single-image methods are unaware of
polarization, and polarization-based methods are constrained by traditional polarization
models. These deficiencies lead to inaccurate polarimetric signatures and physical
inconsistencies in scattering environments. To overcome these limitations, we propose
**DuRP**, a dual-stage physics-embedded learning framework for joint restoration of
scene radiance and polarization information. Specifically, we derive generalized
polarization physics models that relax the ideal assumptions of traditional theory to
provide a more precise foundation for the joint restoration of polarimetric and amplitude
information. We then design a dual-stage neural network to estimate latent physical
parameters through differentiable operators, ensuring that both the polarimetric state
and radiance are accurately recovered. Experimental results show that DuRP achieves
state-of-the-art performance in joint restoration and significantly enhances
polarization-based vision tasks.

## Repository Structure

```text
DuRP/
|-- base/                 # Base trainer/checkpoint utilities
|-- dataset/              # Dataset loaders for train/val/test
|-- model/                # PRNet, IRNet, and full DuRP model definitions
|-- scheduler/            # Warmup and learning-rate schedulers
|-- scripts/
|   |-- make_dataset.py   # Synthetic data generation
|   |-- test_speed.py     # Inference, result saving, latency/FPS reporting
|   `-- calc_metric.py    # PSNR/SSIM evaluation
|-- train/                # Training entry point, options, losses
|-- trainer/              # Stage-specific trainer implementations
|-- utils/                # Logging, metrics, checkpoint, visualization helpers
|-- train_img_reconstruct.sh
|-- train_pol_reconstruct.sh
|-- train_full_stage.sh
`-- test.sh
```

## Installation

The code was developed for CUDA-enabled PyTorch. A minimal environment can be created as follows:

```bash
conda create -n durp python=3.10 -y
conda activate durp

# Install PyTorch following your CUDA version:
# https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install numpy opencv-python tqdm timm tensorboardX albumentations scikit-image polanalyser einops
```

Optional but recommended:

```bash
pip install tensorboard
```

## Data Preparation

DuRP expects generated `.npy` data with the following layout:

```text
datas/SRPG/
|-- train/
|   |-- input/
|   |   |-- I_alpha/      # Four polarization-angle RGB
|   |   `-- m_I/          # Complex observed polarization 
|   `-- target/
|       |-- R/            # Ground-truth scene radiance / S0 image
|       |-- m_A/          # Airlight polarization representation
|       |-- A_inf/        # Atmospheric light
|       |-- m_D/          # Direct-transmission polarization 
|       `-- K/            # Transmission-related target
`-- test/
    `-- ...
```

To synthesize this structure from raw clean polarization data and depth maps, prepare:

```text
raw_data/
|-- train/
|   |-- full_pol/*.npy
|   `-- depth/*.npy
`-- test/
    |-- full_pol/*.npy
    `-- depth/*.npy
```

Then run:

```bash
cd DuRP

python scripts/make_dataset.py \
  --input_dir raw_data \
  --output_dir datas/SRPG \
  --mode train \
  --enlarge_factor 3

python scripts/make_dataset.py \
  --input_dir raw_data \
  --output_dir datas/SRPG \
  --mode test \
  --enlarge_factor 1
```

## Training

All training commands should be run from the `DuRP/` directory.

### Stage 1: Polarization Reconstruction

```bash
python train/train.py \
  --description polarization_reconstruct \
  --arch PRNet \
  --stage pol \
  --loss pol_loss \
  --train_dir ./datas/SRPG/train \
  --val_dir ./datas/SRPG/test \
  --batch_size 1 \
  --train_ps 512 \
  --val_ps 512 \
  --nepoch 400 \
  --lr_initial 1e-4 \
  --weight_decay 1e-4 \
  --gpu_device 0 \
  --save_dir ./experiment/
```

Checkpoints are saved to:

```text
experiment/PRNet_0/checkpoints/
```

### Stage 2: Image Reconstruction

```bash
python train/train.py \
  --description image_reconstruct \
  --arch IRNet \
  --stage img \
  --loss img_loss \
  --train_dir ./datas/SRPG/train \
  --val_dir ./datas/SRPG/test \
  --batch_size 4 \
  --train_ps 512 \
  --val_ps 512 \
  --nepoch 400 \
  --lr_initial 1e-4 \
  --weight_decay 1e-4 \
  --gpu_device 0 \
  --save_dir ./experiment/
```

Checkpoints are saved to:

```text
experiment/IRNet_0/checkpoints/
```

### Stage 3: Full DuRP Fine-Tuning

The full-stage trainer initializes DuRP from pretrained PRNet and IRNet weights:

```bash
python train/train.py \
  --description full_stage \
  --arch DuRP \
  --stage full \
  --loss full_loss \
  --train_dir ./datas/SRPG/train \
  --val_dir ./datas/SRPG/test \
  --PRNet_weights ./experiment/PRNet_0/checkpoints/model_best.pth \
  --IRNet_weights ./experiment/IRNet_0/checkpoints/model_best.pth \
  --batch_size 1 \
  --train_ps 512 \
  --val_ps 512 \
  --nepoch 400 \
  --T_max 32 \
  --lr_initial 1e-5 \
  --eta_min 1e-6 \
  --weight_decay 1e-4 \
  --gpu_device 0 \
  --save_dir ./experiment/
```

## Inference

Run inference and save reconstructed outputs:

```bash
python scripts/test_speed.py \
  --val_dir ./datas/SRPG/test \
  --weights ./experiment/DuRP_0/checkpoints/model_latest.pth \
  --result_dir ./results/SRPG/test \
  --arch DuRP \
  --stage test \
  --gpu_device 0 \
  --mode fp16 \
  --save
```

Supported inference modes:

- `fp32`: standard full-precision inference.
- `fp16`: mixed-precision inference with autocast.
- `fp16_compile`: mixed precision with `torch.compile` enabled; requires PyTorch 2.0+.

Saved results follow this layout:

```text
results/SRPG/test/
|-- S0_R/       # Recovered radiance images, PNG
|-- dolp_R/     # Recovered DoLP, NPY
`-- aolp_R/     # Recovered AoLP, NPY
```

## Evaluation

After inference, compute PSNR and SSIM:

```bash
python scripts/calc_metric.py \
  --gt_dir ./datas/SRPG/test/GT \
  --pred_dir ./results/SRPG/test
```

The script reports metrics for:

- `S0`: reconstructed scene radiance image.
- `DoLP`: degree of linear polarization.
- `AoLP`: angle of linear polarization, using periodic PSNR.

## Reproducibility
Logs and TensorBoard summaries are written under:

```text
experiment/<ARCH>_<ENV>/logs/
experiment/<ARCH>_<ENV>/runs/
```

To resume from a checkpoint:

```bash
python train/train.py \
  --resume \
  --pretrain_weights ./experiment/PRNet_0/checkpoints/model_latest.pth \
  ...
```

## Checkpoints and Datasets
Pretrained checkpoints and processed datasets will be released at:
- Checkpoints: [https://drive.google.com/drive/folders/1ytiAFX0Y3nIlrZdntMB15I7BrAgGudoo?usp=sharing](https://drive.google.com/drive/folders/1ytiAFX0Y3nIlrZdntMB15I7BrAgGudoo?usp=sharing)
- dataset: [![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/yangzhenshuo/DuRP_Dataset)

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{yang2026durp,
  title={DuRP: Dual-Stage Physics-Embedded Learning for Joint Radiance and Polarization Restoration},
  author={Yang, Zhenshuo and He, Qian and Liu, Zhiyuan and Fan, Baojie and Tian, Jiandong},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## License

This project is released for research use. Please add the final license file before public release.

## Acknowledgements

This implementation uses PyTorch and builds on common open-source utilities for image restoration, polarization analysis, and experiment logging. Please also cite the relevant datasets and dependencies when using this code.
