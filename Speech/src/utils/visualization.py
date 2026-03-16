import matplotlib.pyplot as plt
import numpy as np
import os

def plot_attack_comparison(clean_audio, adv_audio, title, save_path):
    """Plot waveform and spectrogram comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Waveforms
    axes[0, 0].plot(clean_audio, color='blue', alpha=0.7)
    axes[0, 0].set_title("Original Waveform")
    axes[0, 1].plot(adv_audio, color='red', alpha=0.7)
    axes[0, 1].set_title("Adversarial Waveform")
    
    # Spectrograms
    axes[1, 0].specgram(clean_audio, Fs=16000)
    axes[1, 0].set_title("Original Spectrogram")
    axes[1, 1].specgram(adv_audio, Fs=16000)
    axes[1, 1].set_title("Adversarial Spectrogram")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_meta_results(results_df, save_dir):
    """Plot overall experiment metrics"""
    # Example: Distribution of WER
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['wer_orig'], bins=20, alpha=0.5, label='Original WER')
    plt.hist(results_df['wer_adv'], bins=20, alpha=0.5, label='Adversarial WER')
    plt.legend()
    plt.title("WER Distribution Comparison")
    plt.savefig(os.path.join(save_dir, "wer_distribution.png"))
    plt.close()
