from typing import List
import torch
import evaluate


# -----------------------------------------------------------------------------------------------
class WordErrorRate:
    def __init__(self):
        self.metric = evaluate.load("wer")

    def __call__(self, predictions: List[str], references: List[str]) -> float:
        return 100 * self.metric.compute(predictions=predictions, references=references)


# -----------------------------------------------------------------------------------------------
class CharErrorRate:
    def __init__(self):
        self.metric = evaluate.load("cer")

    def __call__(self, predictions: List[str], references: List[str]) -> float:
        return 100 * self.metric.compute(predictions=predictions, references=references)


# -----------------------------------------------------------------------------------------------
def compute_snr(adv: torch.Tensor, org: torch.Tensor) -> float:
    # 10*log_10(Power_signal / Power_noise)
    noise = adv - org
    snr = 10 * torch.log10(torch.mean(torch.square(org)) / torch.mean(torch.square(noise)))
    return snr.item()
