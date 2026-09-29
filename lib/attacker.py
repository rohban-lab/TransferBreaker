import torch
from torch import Tensor
from typing import List, Dict
from .metrics import WordErrorRate, compute_snr


# -----------------------------------------------------------------------------------------------
class TrainingAttacker:
    """Momentum PGD used during adversarial fine-tuning.

    The SNR constraint determines a sample-dependent L-infinity radius epsilon_x.
    The initial/update step size is alpha_ratio * epsilon_x for each utterance.
    """

    def __init__(self, alpha_ratio, steps, target_snr, eps_const=1.0, momentum=0.9):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if alpha_ratio <= 0:
            raise ValueError("alpha_ratio must be > 0")
        if eps_const <= 0:
            raise ValueError("eps_const must be > 0")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")

        self.alpha_ratio = alpha_ratio
        self.steps = steps
        self.target_snr = target_snr
        self.eps_const = eps_const
        self.momentum = momentum

    @staticmethod
    def _normalized_input_grads(model, adv_audios: List[Tensor], labels: Tensor):
        """Return L1-normalized dL/dx without accumulating model parameter gradients."""
        loss = model.compute_loss(adv_audios, labels)
        grads = torch.autograd.grad(
            loss,
            adv_audios,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        normalized = []
        for g in grads:
            g = g.detach()
            n = g.norm(p=1)
            normalized.append(g / n if n > 0 else g)
        return loss.detach(), normalized

    def mix_attack(self, base_model, target_model, audios: List[Tensor], labels: Tensor, gamma=0.5, print_logs=True) -> List[Tensor]:
        assert 0 <= gamma <= 1, f"gamma must be between 0 and 1, got {gamma}"

        device = base_model.device
        labels = labels.to(device)
        org_audios = [a.to(device) for a in audios]
        adv_audios = [a.clone().detach().to(device).requires_grad_(True) for a in org_audios]
        grads_momentum = [torch.zeros_like(a, device=device) for a in adv_audios]

        epsilons = compute_epsilons(audios=org_audios, target_snr=self.target_snr, eps_const=self.eps_const)
        alphas = [self.alpha_ratio * eps for eps in epsilons]

        for s in range(self.steps):
            # Algorithm 2 scheduler: t=(j+1)/n, with j zero-indexed here.
            t = (s + 1) / self.steps
            w_b = 1.0 - gamma * t
            w_t = gamma * t

            adv_audios = [a.detach().requires_grad_(True) for a in adv_audios]
            loss_base, grads_base = self._normalized_input_grads(base_model, adv_audios, labels)

            adv_audios = [a.detach().requires_grad_(True) for a in adv_audios]
            loss_target, grads_target = self._normalized_input_grads(target_model, adv_audios, labels)

            with torch.no_grad():
                for i in range(len(adv_audios)):
                    grads_tot = (w_b * grads_base[i]) + (w_t * grads_target[i])
                    grads_momentum[i] = grads_tot + (self.momentum * grads_momentum[i])
                    adv_audios[i].add_(alphas[i] * grads_momentum[i].sign())
                    adv_audios[i].clamp_(org_audios[i] - epsilons[i], org_audios[i] + epsilons[i])
                    adv_audios[i].clamp_(-1, 1)

            loss_tot = (w_b * loss_base.item()) + (w_t * loss_target.item())
            if (s + 1) % 10 == 0 and print_logs:
                print(f"Iteration {s + 1:>{len(str(self.steps))}}/{self.steps} - Loss: {loss_tot:7.4f}")

        return [a.detach() for a in adv_audios]

    def attack(self, model, audios: List[Tensor], labels: Tensor, print_logs=True) -> List[Tensor]:
        device = model.device
        labels = labels.to(device)
        org_audios = [a.to(device) for a in audios]
        adv_audios = [a.clone().detach().to(device).requires_grad_(True) for a in org_audios]
        grads_momentum = [torch.zeros_like(a, device=device) for a in adv_audios]

        epsilons = compute_epsilons(audios=org_audios, target_snr=self.target_snr, eps_const=self.eps_const)
        alphas = [self.alpha_ratio * eps for eps in epsilons]

        for s in range(self.steps):
            adv_audios = [a.detach().requires_grad_(True) for a in adv_audios]
            loss, grads = self._normalized_input_grads(model, adv_audios, labels)

            with torch.no_grad():
                for i in range(len(adv_audios)):
                    grads_momentum[i] = grads[i] + (self.momentum * grads_momentum[i])
                    adv_audios[i].add_(alphas[i] * grads_momentum[i].sign())
                    adv_audios[i].clamp_(org_audios[i] - epsilons[i], org_audios[i] + epsilons[i])
                    adv_audios[i].clamp_(-1, 1)

            if (s + 1) % 10 == 0 and print_logs:
                print(f"Iteration {s + 1:>{len(str(self.steps))}}/{self.steps} - Loss: {loss.item():7.4f}")

        return [a.detach() for a in adv_audios]


# -----------------------------------------------------------------------------------------------
class GeneralAttacker:
    def __init__(self,
                 model,
                 attack_type,
                 alpha_ratio,
                 steps,
                 target_snr,
                 eps_const=1.0,
                 min_acceptable_wer=0.0,
                 max_acceptable_wer=float("inf"),
                 adaptive_alpha=True,
                 alpha_reduction_rate=0.5,
                 momentum=0.9):

        if attack_type != "untargeted":
            raise ValueError("GeneralAttacker currently supports only untargeted attacks")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if alpha_ratio <= 0:
            raise ValueError("alpha_ratio must be > 0")
        if eps_const <= 0:
            raise ValueError("eps_const must be > 0")
        if not 0.0 < alpha_reduction_rate <= 1.0:
            raise ValueError("alpha_reduction_rate must be in (0, 1]")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")

        self.model = model
        self.device = model.device
        self.attack_type = attack_type
        self.alpha_ratio = alpha_ratio
        self.steps = steps
        self.target_snr = target_snr
        self.eps_const = eps_const
        self.min_acceptable_wer = min_acceptable_wer
        self.max_acceptable_wer = max_acceptable_wer
        self.adaptive_alpha = adaptive_alpha
        self.alpha_reduction_rate = alpha_reduction_rate
        self.momentum = momentum
        self.wer = WordErrorRate()

    def attack(self, batch, print_logs=True) -> Dict:
        labels = self.model.extract_labels(batch["transcript"])
        org_audios = [a.to(self.device) for a in batch["audio"]]
        adv_audios = [a.clone().detach().to(self.device).requires_grad_(True) for a in org_audios]
        grads_momentum = [torch.zeros_like(a, device=self.device) for a in adv_audios]

        epsilons = compute_epsilons(audios=org_audios, target_snr=self.target_snr, eps_const=self.eps_const)
        initial_alphas = [self.alpha_ratio * eps for eps in epsilons]
        alpha_scale = 1.0
        batch_size = len(org_audios)

        best_loss = 0.0
        stagnant_steps = 0

        for s in range(self.steps):
            adv_audios = [a.detach().requires_grad_(True) for a in adv_audios]
            loss = self.model.compute_loss(adv_audios, labels)
            grads = torch.autograd.grad(
                loss,
                adv_audios,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

            with torch.no_grad():
                for i in range(batch_size):
                    grad = grads[i].detach()
                    l1_norm = grad.norm(p=1)
                    if l1_norm > 0:
                        grads_momentum[i] = self.momentum * grads_momentum[i] + grad / l1_norm
                    else:
                        grads_momentum[i] = self.momentum * grads_momentum[i]

                    alpha_i = alpha_scale * initial_alphas[i]
                    adv_audios[i].add_(alpha_i * grads_momentum[i].sign())

                    adv_audios[i].clamp_(org_audios[i] - epsilons[i], org_audios[i] + epsilons[i])
                    adv_audios[i].clamp_(-1, 1)

            loss_value = loss.item()
            if loss_value > best_loss:
                best_loss = loss_value
                stagnant_steps = 0
            else:
                stagnant_steps += 1

            if stagnant_steps >= 10 and self.adaptive_alpha:
                alpha_scale *= self.alpha_reduction_rate
                stagnant_steps = 0
                if print_logs:
                    print(f"no improvement for 10 consecutive steps → reducing alpha scale to {alpha_scale:.3f}")

            if (s + 1) % 10 == 0 and print_logs:
                mean_alpha = torch.stack([a.detach() for a in initial_alphas]).mean().item() * alpha_scale
                print(f"Iteration {s + 1:>{len(str(self.steps))}}/{self.steps} - Loss: {loss_value:7.4f} - Mean alpha: {mean_alpha:.2e}")

        accepted = 0
        sum_snr = 0.0
        sum_wer = 0.0
        successful_examples = {"file_name": [], "audio": [], "transcript": [], "prediction": [], "wer": []}
        for i in range(batch_size):
            adv_audio = adv_audios[i].detach()
            org_audio = org_audios[i]

            pred = self.model.transcribe([adv_audio])
            ref = [batch["transcript"][i]]
            wer = self.wer(predictions=pred, references=ref)
            sum_wer += wer
            sum_snr += compute_snr(adv_audio, org_audio)

            if self.min_acceptable_wer <= wer <= self.max_acceptable_wer:
                accepted += 1
                successful_examples["file_name"].append(batch["file_name"][i])
                successful_examples["audio"].append(adv_audio)
                successful_examples["transcript"].append(batch["transcript"][i])
                successful_examples["prediction"].append(pred[0])
                successful_examples["wer"].append(wer)

        if accepted == 0:
            print("attack failed! check the hyperparameters")
        else:
            if self.min_acceptable_wer == 0.0 and self.max_acceptable_wer == float("inf"):
                print(f"mean WER: {sum_wer / batch_size:.2f} , mean SNR: {sum_snr / batch_size:.2f}")
            else:
                print(f"success rate: {(accepted / batch_size) * 100:.2f}% , mean WER: {sum_wer / batch_size:.2f} , mean SNR: {sum_snr / batch_size:.2f}")

        return successful_examples


# -----------------------------------------------------------------------------------------------
def compute_epsilons(audios: List[Tensor], target_snr: float, eps_const: float) -> List[Tensor]:
    power_ratio = 10.0 ** (target_snr / 10.0)
    eps_const = torch.tensor(eps_const, device=audios[0].device)
    eps_list = []

    for signal in audios:
        signal_power = torch.mean(torch.square(signal))
        eps = eps_const * torch.sqrt(signal_power / power_ratio)
        eps_list.append(eps)

    return eps_list
