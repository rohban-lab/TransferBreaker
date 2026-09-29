from lib import *
import torch
import numpy as np
from scipy.io.wavfile import write
from datetime import datetime
import argparse
import warnings
from omegaconf import OmegaConf
import logging
import sys
import os
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()
cfg = OmegaConf.load(args.config)

# -----------------------------------------------------------------------------------------------

set_seed(cfg.seed)
os.makedirs(f"{cfg.data.out_dir}/{cfg.data.clean_split}/audio", exist_ok=True)
os.makedirs(cfg.data.log_dir, exist_ok=True)

# -----------------------------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config_save_path = f"{cfg.data.out_dir}/config_{timestamp}.yaml"
config_save_path_logs = f"{cfg.data.log_dir}/config_{timestamp}.yaml"
OmegaConf.save(cfg, config_save_path)
OmegaConf.save(cfg, config_save_path_logs)
print(f"config saved to: {config_save_path}")
print(f"config saved to: {config_save_path_logs}")

# -----------------------------------------------------------------------------------------------

sys.stdout = Tee(f"{cfg.data.log_dir}/output.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

dataset = load_dataset(dir_path=cfg.data.clean_dir, lang=cfg.lang, split=cfg.data.clean_split, frac=cfg.data.clean_frac, seed=cfg.seed)
print("dataset size:", len(dataset))

# -----------------------------------------------------------------------------------------------

dataloader = create_dataloader(dataset, batch_size=cfg.attacker.batch_size)
print(f"num. of batches: {len(dataloader)}")

# -----------------------------------------------------------------------------------------------

attacker = GeneralAttacker(model=model,
                           attack_type=cfg.attacker.attack_type,
                           alpha_ratio=cfg.attacker.alpha_ratio,
                           steps=cfg.attacker.steps,
                           target_snr=cfg.attacker.target_snr,
                           adaptive_alpha=cfg.attacker.adaptive_alpha)

# -----------------------------------------------------------------------------------------------

start_time = datetime.now()
dataframe = []

for b, batch in enumerate(dataloader):
    print(f"---------- batch: {b + 1} ----------")
    out = attacker.attack(batch)

    for i in range(len(out["audio"])):
        # save adversarial audios
        audio_tensor = out["audio"][i].cpu().squeeze(0).numpy()
        audio_np = np.array(audio_tensor * 32767, dtype=np.int16)
        write(cfg.data.out_dir + f"/{cfg.data.clean_split}/audio/" + out["file_name"][i], 16000, audio_np)
        # add file name and transcript to dataframe
        dataframe.append({"file_name": out["file_name"][i],
                          "transcript": out["transcript"][i],
                          "prediction": out["prediction"][i],
                          "wer": out["wer"][i]})

# save transcripts
df = pd.DataFrame(dataframe, columns=["file_name", "transcript", "prediction", "wer"])
df.to_csv(cfg.data.out_dir + f"/{cfg.data.clean_split}/transcript.csv", index=False, encoding="utf-8-sig")

end_time = datetime.now()
print(f"Execution time: {end_time - start_time}")

tot_mean_wer = np.mean(df["wer"])
print(f"\nTotal mean WER: {tot_mean_wer:.2f}")

# -----------------------------------------------------------------------------------------------
