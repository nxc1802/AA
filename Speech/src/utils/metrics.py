import jiwer
import numpy as np

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate
    """
    # Lowercase and remove basic punctuation for fairer comparison
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    # Simple normalizer
    transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    ref_norm = transform(reference)
    hyp_norm = transform(hypothesis)
    
    if len(ref_norm) == 0:
        return 0.0 if len(hyp_norm) == 0 else 1.0
        
    return jiwer.wer(ref_norm, hyp_norm)

from pesq import pesq
from pystoi import stoi
import torch
from torchmetrics.text import CharErrorRate

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate"""
    cer_metric = CharErrorRate()
    return cer_metric(hypothesis.lower(), reference.lower()).item()

def calculate_lp_norms(clean_audio, perturbed_audio):
    """Calculate L2 and Linf norms of the perturbation"""
    diff = perturbed_audio - clean_audio
    l2 = np.linalg.norm(diff, ord=2)
    linf = np.linalg.norm(diff, ord=np.inf)
    return l2, linf

def calculate_audio_quality(clean_audio, perturbed_audio, fs=16000):
    """Calculate PESQ and STOI"""
    # Ensure audio is properly normalized and typed for PESQ/STOI
    clean = clean_audio.astype(np.float32)
    perturbed = perturbed_audio.astype(np.float32)
    
    # STOI
    d = stoi(clean, perturbed, fs, extended=False)
    
    # PESQ (Narrow-band 'nb' for 16kHz)
    try:
        score_pesq = pesq(fs, clean, perturbed, 'wb')
    except Exception as e:
        print(f"Warning: PESQ calculation failed: {e}")
        score_pesq = 0.0
        
    return score_pesq, d

def calculate_snr(clean_audio, perturbed_audio):
    """
    Calculate Signal-to-Noise Ratio (in dB)
    """
    signal_power = np.sum(clean_audio ** 2)
    noise = perturbed_audio - clean_audio
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
        
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db
