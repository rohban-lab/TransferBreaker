from transformers import WhisperForConditionalGeneration, WhisperProcessor
from .normalizer import Normalizer
import torch
from torch import Tensor
import torchaudio
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Union


# -----------------------------------------------------------------------------------------------
class WhisperWrapper:
    def __init__(self, model_path, lang, device, cache_dir=None, n_mels=None):
        """
        NOTE: model_path can be the address of a local directory or the address of a Hugging Face repository
        """
        self.device = device
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir).eval().to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_path, cache_dir=cache_dir, language=lang, task="transcribe")
        self.tokenizer = self.processor.tokenizer
        if n_mels is None:
            n_mels = getattr(self.model.config, "num_mel_bins", 80)
        self.feature_extractor = TorchWhisperFeatureExtractor(n_mels=n_mels)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=lang, task="transcribe")
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
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            log_spec, attention_mask = self.feature_extractor(audios)
            pred_ids = self.model.generate(input_features=log_spec,
                                           attention_mask=attention_mask,
                                           forced_decoder_ids=self.forced_decoder_ids)

        transcripts = self.tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
        normalized_transcripts = self.normalizer(transcripts)
        return normalized_transcripts
    
    def compute_loss(self, audios: List[Tensor], labels: List[Tensor], return_hidden_states=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        log_spec, attention_mask = self.feature_extractor(audios)
        out = self.model(input_features=log_spec, attention_mask=attention_mask, labels=labels)
        return (out.loss, out.encoder_last_hidden_state) if return_hidden_states else out.loss

    def compute_hidden_states(self, audios: List[Tensor]) -> Tensor:
        log_spec, _ = self.feature_extractor(audios)
        out = self.model.model.encoder(log_spec)
        return out.last_hidden_state

    def extract_labels(self, transcripts: List[str]) -> List[Tensor]:
        labels = self.tokenizer(transcripts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def custom_generate(self, input_features: Tensor, attention_mask: Tensor, max_length=100) -> Tuple[str, Tensor]:
        # NOTE: this method only works for a single input (not a batch)
        with torch.no_grad():
            # prepare prompt tokens
            start_of_transcript_token = [self.model.config.decoder_start_token_id]
            lang_task_no_timestamps_tokens = [token for _, token in self.forced_decoder_ids]

            # prompt_ids: [<|startoftranscript|>, <|lang_id|>, <|task|>, <|notimestamps|>]
            prompt_ids = torch.tensor([start_of_transcript_token + lang_task_no_timestamps_tokens]).to(self.device)

            # compute audio embeddings
            encoder_outputs = self.model.model.encoder(input_features=input_features, attention_mask=attention_mask)

            # prepare for autoregressive loop
            decoder_input_ids = prompt_ids.clone()
            logits_list = []

            # autoregressive loop
            for _ in range(max_length):
                # predict the next_token_logits
                outputs = self.model.model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_outputs.last_hidden_state)
                next_token_logits = self.model.proj_out(outputs.last_hidden_state[:, -1, :])

                # greedy search and add the new token to the last tokens
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)

                # save the logits
                logits_list.append(next_token_logits)

                # check for EOS token
                if next_token_id.item() == self.model.config.eos_token_id:
                    break

            # concatenate all logits
            logits = torch.cat(logits_list, dim=0)

            # decode output
            transcription = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)[0]

            return transcription, logits


# -----------------------------------------------------------------------------------------------
class TorchWhisperFeatureExtractor(nn.Module):
    def __init__(self, n_mels=80, dither=0.0):
        """
        NOTE: in whisper-large-v3 -> n_mels=128
        NOTE: the chunk length is 30s, and unfortunately, in practice, you can’t change it
        NOTE: sample rate is fix at 16000
        """
        super().__init__()
        self.n_fft = 400
        self.hop_length = 160  # each frame -> 10ms
        self.n_mels = n_mels
        self.dither = dither

        self.window = torch.hann_window(self.n_fft)

        # mel filter bank
        self.mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=1 + (self.n_fft // 2),
            f_min=0.0,
            f_max=8000.0,
            n_mels=self.n_mels,
            sample_rate=16000,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveforms: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        waveform: A list of tensors with shape [1, T_i] and values in range of [-1, 1]
        returns: (Mel Spectrogram: [B, n_mels, 3000], Attention Mask: [B, 3000])
        """
        device = waveforms[0].device
        batch_size = len(waveforms)
        org_lengths = [w.shape[-1] for w in waveforms]
        max_len = max(org_lengths)

        # move to device
        window = self.window.to(device)
        mel_fb = self.mel_fb.to(device)

        # pad the input audios
        padded_waveforms = []
        for w in waveforms:
            padding = max_len - w.shape[-1]
            padded_w = F.pad(w, (0, padding), mode='constant', value=0.0)
            padded_waveforms.append(padded_w)
        waveforms = torch.cat(padded_waveforms, dim=0).to(device)  # [B, T_max]

        if self.dither != 0.0:
            waveforms = waveforms + self.dither * torch.randn_like(waveforms)

        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True
        )  # [B, 1 + (n_fft//2), frames]

        # map magnitude to log-mel spectrogram
        mel_spec = torch.matmul(mel_fb.T, stft.abs() ** 2)  # [B, n_mels, frames]
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_per_batch = log_spec.amax(dim=2, keepdim=True)
        log_spec = torch.maximum(log_spec, max_per_batch - 8.0)

        # normalize into ~[-1, 1]
        log_spec = (log_spec + 4.0) / 4.0

        # pad / truncate to 3000 frames (30s)
        frames = log_spec.size(-1)
        if frames > 3000:
            raise ValueError("Number of frames must not exceed 3000 (30s).")
        else:
            log_spec = F.pad(log_spec, (0, 3000 - frames))

        # create attention mask
        attention_mask = torch.zeros(batch_size, 3000, dtype=torch.bool, device=device)
        frame_lengths = [(length // self.hop_length) for length in org_lengths]

        for i, length in enumerate(frame_lengths):
            attention_mask[i, :length+1] = True

        return log_spec, attention_mask
