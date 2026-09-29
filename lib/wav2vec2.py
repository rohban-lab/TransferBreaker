from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from .normalizer import Normalizer
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Union


# -----------------------------------------------------------------------------------------------
class Wav2Vec2Wrapper:
    def __init__(self, model_path, lang, device, cache_dir=None):
        self.device = device
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path, cache_dir=cache_dir).eval().to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path, cache_dir=cache_dir)

        # Ensure consistent CTC loss behavior across different models
        # Some models use 'sum' (e.g., facebook/wav2vec2-large-xlsr-53-polish)
        # Others use 'mean' (e.g., jonatasgrosman/wav2vec2-large-xlsr-53-persian)
        # We normalize to 'mean' for consistent loss scales
        self.model.config.ctc_loss_reduction = "mean"
        self.model.config.ctc_zero_infinity = True
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = TorchWav2Vec2FeatureExtractor()
        self.normalizer = Normalizer(lang=lang)

    def freeze(self):
        self.model.requires_grad_(False)

    def unfreeze(self):
        self.model.requires_grad_(True)

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def training_status(self):
        return self.model.training

    def transcribe(self, audios: List[Tensor]) -> List[str]:
        # Disable autocast for accurate evaluation metrics (match training evaluation behavior)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            input_values, attention_mask = self.feature_extractor(audios)
            logits = self.model(input_values=input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)

        transcripts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        normalized_transcripts = self.normalizer(transcripts)
        return normalized_transcripts

    def compute_loss(self, audios: List[Tensor], labels: Tensor, return_hidden_states=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        input_values, attention_mask = self.feature_extractor(audios)
        out = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=return_hidden_states
        )

        return (out.loss, out.hidden_states[-1]) if return_hidden_states else out.loss

    def compute_hidden_states(self, audios: List[Tensor]) -> Tensor:
        input_values, attention_mask = self.feature_extractor(audios)
        out = self.model(
            input_values=input_values, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        return out.hidden_states[-1]
    
    def extract_labels(self, transcripts: List[str]) -> Tensor:
        # Use tokenizer directly instead of as_target_processor context
        with torch.no_grad():
            labels = self.tokenizer(
                transcripts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).input_ids.to(self.device)
        
        # Replace padding with -100 to ignore in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels


# -----------------------------------------------------------------------------------------------
class TorchWav2Vec2FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, waveforms: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Input: List of tensors [1, T]
        Output: (input_values [B, T_max], attention_mask [B, T_max])
        """
        device = waveforms[0].device
        # Calculate max length for padding
        org_lengths = [w.shape[-1] for w in waveforms]
        max_len = max(org_lengths)

        processed_waveforms = []
        
        for w in waveforms:
            mean = w.mean(dim=-1, keepdim=True)
            var = w.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + 1e-7)  # 1e-7 for numerical stability
            norm_w = (w - mean) / std

            # 2. Pad (Apply padding to the NORMALIZED audio)
            padding = max_len - w.shape[-1]
            
            # Pad with 0.0. Since we normalized to mean 0, padding with 0 is consistent.
            padded_w = F.pad(norm_w, (0, padding), mode='constant', value=0.0)
            processed_waveforms.append(padded_w)
        
        # [B, T_max]
        input_values = torch.cat(processed_waveforms, dim=0)

        # Create attention mask (1 for real audio, 0 for padding)
        attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long, device=device)
        for i, length in enumerate(org_lengths):
            attention_mask[i, :length] = 1

        return input_values, attention_mask
