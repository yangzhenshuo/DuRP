import torch

def myPSNR(tar, prd):
    # judge whether the input is complex
    if torch.is_complex(tar):
        tar = torch.abs(tar)
    if torch.is_complex(prd):
        prd = torch.abs(prd)
    imdff = torch.clamp(prd,0,1) - torch.clamp(tar,0,1)
    rmse = (imdff**2).mean().sqrt()
    if rmse == 0:
        ps = 100 # 如果 MSE 为零，返回100
        return ps  
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)