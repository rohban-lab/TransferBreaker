import torch.nn as nn
import torch
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from .metrics import WordErrorRate, CharErrorRate, compute_snr
from .attacker import TrainingAttacker
from .utils import setup_logger
import numpy as np
import copy
import gc
import math


# ----------------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, dataset, seed, out_dir, log_dir):
        self.model = model
        self.dataset = dataset
        self.seed = seed
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.compute_wer = WordErrorRate()
        self.compute_cer = CharErrorRate()

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def train(self,
              steps,
              alpha_ratio,
              targeted_snr,
              lr,
              epochs,
              batch_size,
              grad_accumulation_steps=1,
              dataloader_num_workers=4,
              bf16=True,
              gradient_checkpointing=False,
              warmup_ratio=0.1,
              objective_mode="default",
              lambda_adv=4.0,
              lambda_ljr=2.5,
              hybrid_prob=0.5,
              beta=0.625,
              gamma=0.5,
              p_clean=0.2,
              p_adv_base=0.4,
              p_adv_target=0.0,
              p_adv_mix=0.4,
              adv_mix_target_update=100,
              log_interval=25,
              checkpoint_save_limit=4):

        if objective_mode not in {"default", "efficient"}:
            raise ValueError("objective_mode must be either 'default' or 'efficient'")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if alpha_ratio <= 0:
            raise ValueError("alpha_ratio must be > 0")
        if lambda_adv < 0 or lambda_ljr < 0 or beta < 0:
            raise ValueError("lambda_adv, lambda_ljr, and beta must be non-negative")
        if not 0.0 <= hybrid_prob <= 1.0:
            raise ValueError("hybrid_prob must be in [0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if adv_mix_target_update < 1:
            raise ValueError("adv_mix_target_update must be >= 1")
        if log_interval < 1:
            raise ValueError("log_interval must be >= 1")
        if objective_mode == "efficient":
            probabilities = (p_clean, p_adv_base, p_adv_target, p_adv_mix)
            if any(p < 0.0 or p > 1.0 for p in probabilities):
                raise ValueError("Efficient-mode branch probabilities must each be in [0, 1]")
            if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
                raise ValueError("Efficient-mode branch probabilities must sum to 1")

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.out_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=grad_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            fp16=False,
            bf16=bf16,
            fp16_full_eval=False,
            bf16_full_eval=False,
            dataloader_num_workers=dataloader_num_workers,
            eval_strategy="epoch",
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            generation_max_length=225,
            report_to=["tensorboard"],
            logging_strategy="steps",
            logging_steps=log_interval,
            save_strategy="epoch",
            push_to_hub=False,
            save_total_limit=checkpoint_save_limit,
            seed=self.seed,
            data_seed=self.seed,
            remove_unused_columns=False,
        )

        speech_data_collator = SpeechDataCollator(
            tokenizer=self.model.tokenizer,
            feature_extractor=self.model.feature_extractor,
        )

        trainer = CustomSeq2SeqTrainer(
            steps=steps,
            alpha_ratio=alpha_ratio,
            targeted_snr=targeted_snr,
            objective_mode=objective_mode,
            lambda_adv=lambda_adv,
            lambda_ljr=lambda_ljr,
            hybrid_prob=hybrid_prob,
            beta=beta,
            gamma=gamma,
            p_clean=p_clean,
            p_adv_base=p_adv_base,
            p_adv_target=p_adv_target,
            p_adv_mix=p_adv_mix,
            adv_mix_target_update=adv_mix_target_update,
            input_model=self.model,
            log_dir=self.log_dir,
            log_interval=log_interval,
            args=training_args,
            model=self.model.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=speech_data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.model.processor,
        )

        trainer.train()

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = self.model.tokenizer.pad_token_id

        pred_str = self.model.tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
        label_str = self.model.tokenizer.batch_decode(label_ids.tolist(), skip_special_tokens=True)
        normalized_pred_str = self.model.normalizer(pred_str)
        normalized_label_str = self.model.normalizer(label_str)

        # Paper results use macro-average utterance-level WER/CER.
        wer_scores = [
            self.compute_wer(predictions=[pred], references=[ref])
            for pred, ref in zip(normalized_pred_str, normalized_label_str)
        ]
        cer_scores = [
            self.compute_cer(predictions=[pred], references=[ref])
            for pred, ref in zip(normalized_pred_str, normalized_label_str)
        ]

        return {
            "wer": sum(wer_scores) / len(wer_scores),
            "cer": sum(cer_scores) / len(cer_scores),
        }


# ----------------------------------------------------------------------------------
@dataclass
class SpeechDataCollator:
    tokenizer: Any
    feature_extractor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audios = []
        transcripts = []
        for feature in features:
            audios.append(torch.tensor(feature["audio"], dtype=torch.float32).unsqueeze(0))
            transcripts.append(feature["transcript"])

        input_features, attention_mask = self.feature_extractor(audios)

        labels = [{"input_ids": feature["labels"]} for feature in features]
        labels = self.tokenizer.pad(labels, return_tensors="pt")
        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)

        return {
            "audio": audios,
            "transcript": transcripts,
            "input_features": input_features,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ----------------------------------------------------------------------------------
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self,
                 steps,
                 alpha_ratio,
                 targeted_snr,
                 objective_mode,
                 lambda_adv,
                 lambda_ljr,
                 hybrid_prob,
                 beta,
                 gamma,
                 p_clean,
                 p_adv_base,
                 p_adv_target,
                 p_adv_mix,
                 adv_mix_target_update,
                 input_model,
                 log_dir,
                 log_interval,
                 **kwargs):

        super().__init__(**kwargs)
        self.steps = steps
        self.alpha_ratio = alpha_ratio
        self.targeted_snr = targeted_snr
        self.objective_mode = objective_mode
        self.lambda_adv = lambda_adv
        self.lambda_ljr = lambda_ljr
        self.hybrid_prob = hybrid_prob
        self.beta = beta
        self.gamma = gamma
        self.p_clean = p_clean
        self.p_adv_base = p_adv_base
        self.p_adv_target = p_adv_target
        self.p_adv_mix = p_adv_mix
        self.adv_mix_target_update = adv_mix_target_update
        self.target_model = input_model
        self.device = self.target_model.device
        self.log_interval = log_interval

        # Default mode always needs the base model. In efficient mode, the base
        # model is only required by Base-AFT/HybridGrad branches (and their LJR
        # term). Clean-only FT and target-only Standard-AFT therefore avoid an
        # unnecessary second model copy.
        self.adv_base_en = objective_mode == "default" or bool(p_adv_base != 0.0)
        self.adv_mix_en = objective_mode == "default" or bool(p_adv_mix != 0.0)
        needs_base_model = self.adv_base_en or self.adv_mix_en

        self.base_model = None
        if needs_base_model:
            self.base_model = copy.deepcopy(self.target_model)
            self.base_model.eval()
            self.base_model.freeze()
            self.base_model.zero_grad()

        self.compute_wer = WordErrorRate()
        self.compute_cer = CharErrorRate()
        self.mse = nn.MSELoss()
        self.count = 0
        self.base_attack_log = False
        self.mix_attack_log = False
        self.target_attack_log = False

        # Only used when adv_mix_target_update > 1. A value of 1 uses the
        # current target model directly, avoiding deepcopy/cache-refresh cost.
        self.frozen_target_model = None
        self.frozen_target_step = None

        self.attacker = TrainingAttacker(
            alpha_ratio=self.alpha_ratio,
            steps=self.steps,
            target_snr=self.targeted_snr,
        )

        self.adv_loss_logger = setup_logger("adv_loss", log_dir)
        self.clean_loss_logger = setup_logger("clean_loss", log_dir)
        self.base_attack_logger = setup_logger("base_attack", log_dir)
        self.target_attack_logger = setup_logger("target_attack", log_dir)
        self.mix_attack_logger = setup_logger("mix_attack", log_dir)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        org_audios = [org_audio.to(self.device) for org_audio in inputs.get("audio")]
        transcripts = inputs.get("transcript")
        labels = inputs.get("labels").to(self.device)

        if self.count % self.log_interval == 0:
            self.base_attack_log = True
            self.mix_attack_log = True
            self.target_attack_log = True

        if self.objective_mode == "default":
            loss = self._compute_default_objective(org_audios, transcripts, labels)
        else:
            loss = self._compute_efficient_objective(org_audios, transcripts, labels)

        self.count += 1
        return loss

    def _compute_default_objective(self, org_audios, transcripts, labels):
        """Explicit Eq. 17-style objective: Lclean + λ1 Ladv + λ2 LLJR."""
        use_hybrid = np.random.uniform(0, 1) < self.hybrid_prob
        if use_hybrid:
            adv_audios, mix_target = self._generate_hybrid_attack(org_audios, labels)
            if self.mix_attack_log:
                self.mix_attack_log = False
                wer_b, cer_b, snr_b = self._compute_metrics(self.base_model, adv_audios, org_audios, transcripts)
                wer_t, cer_t, snr_t = self._compute_metrics(mix_target, adv_audios, org_audios, transcripts)
                self.mix_attack_logger.info(
                    f"base: WER: {wer_b:6.2f} % , CER: {cer_b:6.2f} % , SNR: {snr_b:6.2f} dB  |  "
                    f"target: WER: {wer_t:6.2f} % , CER: {cer_t:6.2f} % , SNR: {snr_t:6.2f} dB"
                )
        else:
            adv_audios = self.attacker.attack(
                model=self.base_model,
                audios=org_audios,
                labels=labels,
                print_logs=False,
            )
            if self.base_attack_log:
                self.base_attack_log = False
                wer, cer, snr = self._compute_metrics(self.base_model, adv_audios, org_audios, transcripts)
                self.base_attack_logger.info(f"WER: {wer:6.2f} % , CER: {cer:6.2f} % , SNR: {snr:6.2f} dB")

        # Compute the trainable losses only after perturbation generation to avoid
        # retaining the clean forward graph throughout the PGD loop.
        clean_loss = self.target_model.compute_loss(org_audios, labels)
        adv_loss, ljr_loss = self._compute_adv_components(adv_audios, org_audios, labels)
        total_loss = clean_loss + (self.lambda_adv * adv_loss) + (self.lambda_ljr * ljr_loss)

        self.clean_loss_logger.info(f"{self.count:6} , clean loss: {clean_loss.item():9.5f}")
        self.adv_loss_logger.info(
            f"{self.count:6} , adv loss: {adv_loss.item():9.5f} , LJR: {ljr_loss.item():9.5f} , "
            f"total: {total_loss.item():9.5f}"
        )
        return total_loss

    def _compute_efficient_objective(self, org_audios, transcripts, labels):
        """Compute-saving stochastic estimator of the full objective."""
        random_num = np.random.uniform(0, 1)

        if random_num < self.p_clean:
            loss = self.target_model.compute_loss(org_audios, labels)
            self.clean_loss_logger.info(f"{self.count:6} , CE loss: {loss.item():9.5f}")

        elif random_num < self.p_clean + self.p_adv_base:
            adv_audios = self.attacker.attack(
                model=self.base_model,
                audios=org_audios,
                labels=labels,
                print_logs=False,
            )
            if self.base_attack_log:
                self.base_attack_log = False
                wer, cer, snr = self._compute_metrics(self.base_model, adv_audios, org_audios, transcripts)
                self.base_attack_logger.info(f"WER: {wer:6.2f} % , CER: {cer:6.2f} % , SNR: {snr:6.2f} dB")
            loss = self._compute_efficient_adv_loss(adv_audios, org_audios, labels)

        elif random_num < self.p_clean + self.p_adv_base + self.p_adv_target:
            was_training = self.target_model.training_status()
            self.target_model.eval()
            adv_audios = self.attacker.attack(
                model=self.target_model,
                audios=org_audios,
                labels=labels,
                print_logs=False,
            )
            if was_training:
                self.target_model.train()

            if self.target_attack_log:
                self.target_attack_log = False
                wer, cer, snr = self._compute_metrics(self.target_model, adv_audios, org_audios, transcripts)
                self.target_attack_logger.info(f"WER: {wer:6.2f} % , CER: {cer:6.2f} % , SNR: {snr:6.2f} dB")

            # Standard-AFT branch intentionally uses task loss only.
            loss = self.target_model.compute_loss(adv_audios, labels)

        else:
            adv_audios, mix_target = self._generate_hybrid_attack(org_audios, labels)
            if self.mix_attack_log:
                self.mix_attack_log = False
                wer_b, cer_b, snr_b = self._compute_metrics(self.base_model, adv_audios, org_audios, transcripts)
                wer_t, cer_t, snr_t = self._compute_metrics(mix_target, adv_audios, org_audios, transcripts)
                self.mix_attack_logger.info(
                    f"base: WER: {wer_b:6.2f} % , CER: {cer_b:6.2f} % , SNR: {snr_b:6.2f} dB  |  "
                    f"target: WER: {wer_t:6.2f} % , CER: {cer_t:6.2f} % , SNR: {snr_t:6.2f} dB"
                )
            loss = self._compute_efficient_adv_loss(adv_audios, org_audios, labels)

        return loss

    def _generate_hybrid_attack(self, org_audios, labels):
        mix_target = self._get_mix_target_model()

        # adv_mix_target_update=1 uses the live target model directly. Temporarily
        # switch to eval mode for attack generation, then restore training mode.
        is_live_target = mix_target is self.target_model
        was_training = self.target_model.training_status() if is_live_target else False
        if is_live_target:
            self.target_model.eval()

        adv_audios = self.attacker.mix_attack(
            base_model=self.base_model,
            target_model=mix_target,
            audios=org_audios,
            labels=labels,
            gamma=self.gamma,
            print_logs=False,
        )

        if is_live_target and was_training:
            self.target_model.train()

        return adv_audios, mix_target

    def _get_mix_target_model(self):
        """Return live target (interval=1) or a periodically refreshed frozen snapshot.

        Snapshot intervals are measured in optimizer steps (`self.state.global_step`),
        so gradient accumulation does not change the refresh cadence.
        """
        if self.adv_mix_target_update == 1:
            return self.target_model

        current_step = int(self.state.global_step)
        needs_refresh = (
            self.frozen_target_model is None
            or self.frozen_target_step is None
            or (current_step - self.frozen_target_step) >= self.adv_mix_target_update
        )

        if needs_refresh:
            if self.frozen_target_model is not None:
                self.frozen_target_model.model.cpu()
                del self.frozen_target_model.model
                del self.frozen_target_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.frozen_target_model = copy.deepcopy(self.target_model)
            self.frozen_target_model.eval()
            self.frozen_target_model.freeze()
            self.frozen_target_model.zero_grad()
            self.frozen_target_step = current_step

        return self.frozen_target_model

    def _compute_adv_components(self, adv_audios, org_audios, labels):
        adv_loss, target_hidden_states = self.target_model.compute_loss(
            adv_audios,
            labels,
            return_hidden_states=True,
        )
        with torch.no_grad():
            base_hidden_states = self.base_model.compute_hidden_states(org_audios)
        ljr_loss = self.mse(target_hidden_states, base_hidden_states)
        return adv_loss, ljr_loss

    def _compute_efficient_adv_loss(self, adv_audios, org_audios, labels):
        adv_loss, target_hidden_states = self.target_model.compute_loss(
            adv_audios,
            labels,
            return_hidden_states=True,
        )
        if self.beta:
            with torch.no_grad():
                base_hidden_states = self.base_model.compute_hidden_states(org_audios)
            ljr_loss = self.mse(target_hidden_states, base_hidden_states)
        else:
            ljr_loss = torch.tensor(0.0, device=adv_loss.device)

        loss = adv_loss + (self.beta * ljr_loss)
        self.adv_loss_logger.info(
            f"{self.count:6} , adv loss: {adv_loss.item():9.5f} , LJR: {ljr_loss.item():9.5f} , total: {loss.item():9.5f}"
        )
        return loss

    def _compute_metrics(self, model, adv_audios, org_audios, transcripts):
        # Logging metrics should be deterministic even when the live target is
        # currently in training mode. Restore the original mode afterwards.
        was_training = model.training_status()
        model.eval()
        try:
            pred = model.transcribe(adv_audios)
        finally:
            if was_training:
                model.train()

        wer = self.compute_wer(predictions=pred, references=transcripts)
        cer = self.compute_cer(predictions=pred, references=transcripts)
        snr = np.mean([compute_snr(adv, org) for adv, org in zip(adv_audios, org_audios)])
        return wer, cer, snr

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        is_ctc_model = "CTC" in model.__class__.__name__

        if is_ctc_model:
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                model_dtype = next(model.parameters()).dtype
                if model_dtype != torch.float32:
                    model = model.float()

                input_features = inputs.get("input_features").to(self.device).float()
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = inputs.get("labels")
                if labels is not None:
                    labels = labels.to(self.device)

                outputs = model(
                    input_values=input_features,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss if labels is not None else None

                if prediction_loss_only:
                    return loss, None, None

                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=-1)
                return loss, pred_ids, labels
        else:
            evaluation_keys = ["input_features", "labels", "attention_mask"]
            filtered_inputs = {k: v for k, v in inputs.items() if k in evaluation_keys}

            return super().prediction_step(
                model=model,
                inputs=filtered_inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                **gen_kwargs,
            )
