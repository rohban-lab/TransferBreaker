from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from .normalizer import Normalizer
from .wav2vec2 import TorchWav2Vec2FeatureExtractor
import torch
from torch import Tensor
from typing import List, Tuple, Union
import os


# ISO 639-1 (2-letter) to ISO 639-3 (3-letter) mapping for MMS models
LANG_CODE_MAPPING = {
    "pl": "pol",  # Polish
    "pt": "por",  # Portuguese
    "ar": "ara",  # Arabic
    "fi": "fin",  # Finnish
    "vi": "vie",  # Vietnamese
}


# -----------------------------------------------------------------------------------------------
class MMSWrapper:
    def __init__(self, model_path, lang, device, cache_dir=None):
        """
        NOTE: model_path can be the address of a local directory or the address of a Hugging Face repository
        """
        self.device = device
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path, cache_dir=cache_dir).eval().to(self.device)
        # Ensure consistent CTC loss behavior across different models
        self.model.config.ctc_loss_reduction = "mean"
        self.model.config.ctc_zero_infinity = True
        self.processor = Wav2Vec2Processor.from_pretrained(model_path, cache_dir=cache_dir)
        
        # Convert language code to ISO 639-3 format for MMS
        mms_lang_code = LANG_CODE_MAPPING.get(lang, lang)
        
        # MMS models require loading the language-specific adapter
        # Skip if using a fine-tuned checkpoint (adapter weights already merged)
        self.processor.tokenizer.set_target_lang(mms_lang_code)
        adapter_file = os.path.join(model_path, f"adapter.{mms_lang_code}.bin")
        if os.path.isfile(adapter_file) or not os.path.isdir(model_path):
            # Load adapter if it exists locally, or if model_path is a HuggingFace repo
            self.model.load_adapter(mms_lang_code)
        
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
