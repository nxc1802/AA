import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class DifferentiableWhisperFeatureExtractor(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.n_fft = 400
        self.hop_length = 160
        self.chunk_length = 30
        self.sample_rate = 16000
        self.n_samples = self.chunk_length * self.sample_rate
        
        # Extract the mel filters used by HuggingFace's processor
        mel_filters = processor.feature_extractor.mel_filters
        self.register_buffer("mel_filters", torch.tensor(mel_filters, dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, audio):
        # audio shape: (batch_size, sequence_length)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
            
        # Pad or trim to EXACTLY 30 seconds (16000 * 30 = 480000 samples)
        if audio.shape[-1] < self.n_samples:
            padding = self.n_samples - audio.shape[-1]
            audio = F.pad(audio, (0, padding))
        elif audio.shape[-1] > self.n_samples:
            audio = audio[..., :self.n_samples]

        # Calculate STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )
        # Power spectrogram
        magnitudes = stft.abs() ** 2
        # Whisper drops the last frame
        magnitudes = magnitudes[:, :, :-1]
        
        # Apply Mel filterbank
        # HF mel_filters shape is usually [201, n_mels] for Whisper
        # magnitudes shape is [batch_size, 201, time]
        mel_spec = torch.matmul(magnitudes.transpose(1, 2), self.mel_filters).transpose(1, 2)
        
        # Log10 and dynamic range compression
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(1, 2), keepdim=True) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        
        return log_spec

class WhisperModelWrapper:
    def __init__(self, model_id="openai/whisper-tiny", device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        # Performance counters
        self.forward_count = 0
        self.backward_count = 0
        
        print(f"[*] Initializing WhisperModelWrapper on device: {self.device}")
            
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        self.diff_feature_extractor = DifferentiableWhisperFeatureExtractor(self.processor).to(self.device)
        self.diff_feature_extractor.eval()
        
    def transcribe(self, audio_array):
        # audio_array is expected to be a 1D numpy array or torch tensor at 16kHz
        # For standard non-grad transcription, we can use the official processor
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs["input_features"])
            
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
        
    def extract_features(self, audio_tensor):
        """Differentiable feature extraction from 1D waveform"""
        return self.diff_feature_extractor(audio_tensor)
        
    def get_loss(self, audio_waveform, target_text):
        self.forward_count += 1
        """
        Used for adversarial attacks. Needs gradients enabled on audio_tensor.
        """
        input_features = self.extract_features(audio_waveform)
        labels = self.processor(text=target_text, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.model(input_features, labels=labels)
        return outputs.loss, input_features
