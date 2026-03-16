import torch

import torch

def fgsm_attack(model_wrapper, audio_tensor, target_text, epsilon=0.005):
    """
    Standard FGSM, now acting on 1D waveform.
    """
    original_audio = audio_tensor.clone().detach().to(model_wrapper.device)
    perturbed_audio = original_audio.clone().requires_grad_(True)
    
    loss, _ = model_wrapper.get_loss(perturbed_audio, target_text)
    
    model_wrapper.model.zero_grad()
    loss.backward()
    model_wrapper.backward_count += 1
    
    with torch.no_grad():
        perturbed_audio = perturbed_audio + epsilon * perturbed_audio.grad.sign()
        perturbed_audio = torch.clamp(perturbed_audio, min=-1.0, max=1.0)
        
    return perturbed_audio.cpu().numpy().squeeze()

def pgd_attack(model_wrapper, audio_tensor, target_text, epsilon=0.005, alpha=None, num_iter=200):
    """
    Projected Gradient Descent (PGD) attack natively on the 1D audio waveform.
    Uses num_iter=200 by default (as per the referenced paper).
    """
    if alpha is None:
        alpha = epsilon * 0.1 # standard PGD step size rule of thumb
        
    original_audio = audio_tensor.clone().detach().to(model_wrapper.device)
    perturbed_audio = original_audio.clone().requires_grad_(True)
    
    labels = model_wrapper.processor(text=target_text, return_tensors="pt").input_ids.to(model_wrapper.device)
    
    for i in range(num_iter):
        perturbed_audio.requires_grad = True
        
        # 1. Forward pass (includes differentiable 1D -> Mel conversion)
        model_wrapper.forward_count += 1
        input_features = model_wrapper.extract_features(perturbed_audio)
        outputs = model_wrapper.model(input_features, labels=labels)
        loss = outputs.loss
        
        # 2. Backward pass to 1D waveform
        model_wrapper.model.zero_grad()
        loss.backward()
        model_wrapper.backward_count += 1
        
        # 3. Update waveform
        with torch.no_grad():
            adv_audio = perturbed_audio + alpha * perturbed_audio.grad.sign()
            
            # Projection: Clamp back to L_inf epsilon ball around original audio
            eta = torch.clamp(adv_audio - original_audio, min=-epsilon, max=epsilon)
            perturbed_audio = original_audio + eta
            
            # Clamp to valid audio range (-1.0 to 1.0)
            perturbed_audio = torch.clamp(perturbed_audio, min=-1.0, max=1.0)
            perturbed_audio = perturbed_audio.detach()
            
    return perturbed_audio.cpu().numpy().squeeze()
