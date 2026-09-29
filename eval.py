from lib import *
import torch
import argparse
from prettytable import PrettyTable
from omegaconf import OmegaConf
from datetime import datetime
import warnings
import logging
import os
import sys
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()
cfg = OmegaConf.load(args.config)

# -----------------------------------------------------------------------------------------------

set_seed(cfg.seed)
os.makedirs(cfg.data.log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config_save_path = f"{cfg.data.log_dir}/config_{timestamp}.yaml"
OmegaConf.save(cfg, config_save_path)

sys.stdout = Tee(f"{cfg.data.log_dir}/output.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"config saved to: {config_save_path}")
print(f"selected device: {device}")

# -----------------------------------------------------------------------------------------------

if cfg.model_type == "whisper":
    model = WhisperWrapper(model_path=cfg.model.path, lang=cfg.lang, device=device, cache_dir=cfg.model.cache_dir, n_mels=cfg.model.get("n_mels", None))
elif cfg.model_type == "wav2vec2":
    model = Wav2Vec2Wrapper(model_path=cfg.model.path, lang=cfg.lang, device=device, cache_dir=cfg.model.cache_dir)
elif cfg.model_type == "mms":
    model = MMSWrapper(model_path=cfg.model.path, lang=cfg.lang, device=device, cache_dir=cfg.model.cache_dir)
else:
    raise ValueError(f"model type '{cfg.model_type}' is not available")

# -----------------------------------------------------------------------------------------------

dataset = load_dataset(dir_path=cfg.data.eval_dir, lang=cfg.lang, split=cfg.data.eval_split, frac=cfg.data.eval_frac, seed=cfg.seed)
print("eval size:", len(dataset))

# -----------------------------------------------------------------------------------------------

wer = WordErrorRate()
cer = CharErrorRate()


def add_wer_func(batch):
    audios = []
    for audio in batch["audio"]:
        audios.append(torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device))

    predictions = model.transcribe(audios)

    # Normalize references the same way training evaluation does:
    # tokenize -> decode -> normalize (to handle ZWNJ loss during tokenization)
    # ---------------------------------------------------------------
    # The reason for this is that the text after and before (tokenization + decoding) may differ in some languages (e.g., Persian)
    # Because our model is trained on tokenized labels it is fair to evaluate on similarly processed references
    # ---------------------------------------------------------------
    references = []
    for transcript in batch["transcript"]:
        token_ids = model.tokenizer(transcript).input_ids
        decoded = model.tokenizer.decode(token_ids, skip_special_tokens=True)
        normalized_ref = model.normalizer([decoded])[0]
        references.append(normalized_ref)

    # per-sample metrics (for CSV output)
    wers = [round(wer(predictions=[pred], references=[ref]), 2) for pred, ref in zip(predictions, references)]
    cers = [round(cer(predictions=[pred], references=[ref]), 2) for pred, ref in zip(predictions, references)]

    return {"wer": wers, "cer": cers, "prediction": predictions}


dataset = dataset.map(add_wer_func, batched=True, batch_size=cfg.batch_size)

# -----------------------------------------------------------------------------------------------

table = PrettyTable()
table.field_names = ["metric", "mean", "median"]

# sentence-level average metrics
mean_wer = np.mean(dataset["wer"])
mean_cer = np.mean(dataset["cer"])

median_wer = np.median(dataset["wer"])
median_cer = np.median(dataset["cer"])

table.add_row(["WER", f"{mean_wer:.2f}", f"{median_wer:.2f}"])
table.add_row(["CER", f"{mean_cer:.2f}", f"{median_cer:.2f}"])

print(table)

# -----------------------------------------------------------------------------------------------

df = dataset.select_columns(['file_name', 'transcript', 'prediction', 'wer', 'cer']).to_pandas()
df.to_csv(f"{cfg.data.log_dir}/transcript.csv", index=False, encoding="utf-8-sig")

# -----------------------------------------------------------------------------------------------
