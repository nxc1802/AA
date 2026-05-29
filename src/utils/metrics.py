import torch
import numpy as np

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

def calculate_ssim(img1, img2, window_size=11):
    # Simple SSIM implementation for PyTorch
    # Based on: https://github.com/Po-Hsun-Su/pytorch-ssim
    import torch.nn.functional as F
    
    def gaussian(window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size).float() - window_size//2)**2 / (2 * sigma**2))
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def get_metrics(model, images, labels, adv_images, device):
    """
    Calculate comprehensive metrics for a batch of adversarial images.
    Returns: Acc, ASR, L0, L2, Linf, SSIM, PSNR
    """
    model.eval()
    with torch.no_grad():
        # Correctly classified images in clean batch
        outputs_clean = model(images.to(device))
        _, pred_clean = torch.max(outputs_clean, 1)
        correct_idx = (pred_clean == labels.to(device))
        
        # Predicted labels for adversarial images
        outputs_adv = model(adv_images.to(device))
        _, pred_adv = torch.max(outputs_adv, 1)
        
        # 1. Adversarial Accuracy
        acc_adv = (pred_adv == labels.to(device)).float().mean().item()
        
        # 2. ASR (Attack Success Rate)
        if correct_idx.sum() > 0:
            asr = (pred_adv[correct_idx] != labels.to(device)[correct_idx]).float().mean().item()
        else:
            asr = 0.0
            
        # 3. Norms
        diff = (adv_images - images).abs()
        # L0: Number of changed pixels per image (avg across batch)
        # Pixel is changed if any channel difference > 1e-4
        l0_per_image = (diff.max(dim=1)[0] > 1e-4).float().view(diff.size(0), -1).sum(dim=1)
        avg_l0 = l0_per_image.mean().item()
        
        diff_flat = diff.view(diff.size(0), -1)
        l2_norm = torch.norm(diff_flat, p=2, dim=1).mean().item()
        linf_norm = torch.norm(diff_flat, p=float('inf'), dim=1).mean().item()
        
        # 4. Image Quality
        psnr = calculate_psnr(images, adv_images)
        ssim = calculate_ssim(images, adv_images)
        
        # 5. Sparsity (ratio of pixels NOT changed)
        num_pixels = images.size(2) * images.size(3)
        sparsity = 1.0 - (avg_l0 / num_pixels)
        
    return {
        'acc': acc_adv,
        'asr': asr,
        'l0': avg_l0,
        'sparsity': sparsity,
        'l2': l2_norm,
        'linf': linf_norm,
        'psnr': psnr,
        'ssim': ssim
    }
