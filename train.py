from lib import *
import torch
import argparse
from omegaconf import OmegaConf
from datasets import DatasetDict
import warnings
import logging
import os
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()
cfg = OmegaConf.load(args.config)

# -----------------------------------------------------------------------------------------------

os.makedirs(cfg.data.out_dir, exist_ok=True)
os.makedirs(cfg.data.log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config_save_path = f"{cfg.data.out_dir}/config_{timestamp}.yaml"
config_save_path_logs = f"{cfg.data.log_dir}/config_{timestamp}.yaml"
OmegaConf.save(cfg, config_save_path)
OmegaConf.save(cfg, config_save_path_logs)
print(f"config saved to: {config_save_path}")
print(f"config saved to: {config_save_path_logs}")

# -----------------------------------------------------------------------------------------------

set_seed(cfg.seed)
if cfg.float32_matmul_precision:
    if cfg.float32_matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("float32_matmul_precision must be one of: highest, high, medium")
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
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

dataset = DatasetDict()

dataset["train"] = load_dataset(dir_path=cfg.data.clean_dir,
                                lang=cfg.lang,
                                split=cfg.data.clean_split,
                                frac=cfg.data.train_frac,
                                seed=cfg.seed)

dataset["test"] = load_dataset(dir_path=cfg.data.eval_dir,
                               lang=cfg.lang,
                               split=cfg.data.eval_split,
                               frac=cfg.data.eval_frac,
                               seed=cfg.seed)

print("train size:", len(dataset["train"]), " , test size:", len(dataset["test"]))

# -----------------------------------------------------------------------------------------------


def extract_labels_fn(batch):
    batch["labels"] = model.tokenizer(batch["transcript"]).input_ids
    return batch


dataset = dataset.map(extract_labels_fn, batched=True, batch_size=32)

# -----------------------------------------------------------------------------------------------

trainer = Trainer(model, dataset, cfg.seed, cfg.data.out_dir, cfg.data.log_dir)

trainer.train(steps=cfg.trainer.steps,
              alpha_ratio=cfg.trainer.alpha_ratio,
              targeted_snr=cfg.trainer.targeted_snr,
              lr=cfg.trainer.lr,
              epochs=cfg.trainer.epochs,
              batch_size=cfg.trainer.batch_size,
              grad_accumulation_steps=cfg.trainer.grad_accumulation_steps,
              dataloader_num_workers=cfg.trainer.dataloader_num_workers,
              bf16=cfg.trainer.bf16,
              gradient_checkpointing=cfg.trainer.gradient_checkpointing,
              warmup_ratio=cfg.trainer.warmup_ratio,
              objective_mode=cfg.trainer.objective_mode,
              lambda_adv=cfg.trainer.lambda_adv,
              lambda_ljr=cfg.trainer.lambda_ljr,
              hybrid_prob=cfg.trainer.hybrid_prob,
              beta=cfg.trainer.beta,
              gamma=cfg.trainer.gamma,
              p_clean=cfg.trainer.p_clean,
              p_adv_base=cfg.trainer.p_adv_base,
              p_adv_target=cfg.trainer.p_adv_target,
              p_adv_mix=cfg.trainer.p_adv_mix,
              adv_mix_target_update=cfg.trainer.adv_mix_target_update,
              log_interval=cfg.trainer.log_interval,
              checkpoint_save_limit=cfg.trainer.checkpoint_save_limit)

# -----------------------------------------------------------------------------------------------
